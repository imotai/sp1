use sp1_core_machine::{io::SP1Stdin, recursion::SP1RecursionProof};
use sp1_primitives::Elf;
use sp1_prover::{InnerSC, OuterSC, SP1CoreProof, SP1VerifyingKey};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Weak},
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::UnixStream,
    sync::Mutex,
};

use crate::{
    api::{Request, Response},
    pk::CudaProvingKey,
    CudaClientError,
};

/// The global client to be shared, if other clients still exist (like in a proving key.)
static CLIENT: LazyLock<Mutex<HashMap<u32, Weak<CudaClientInner>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// A client that reads and writes length delimited [`Request`] messages to the server.
#[derive(Clone)]
pub(crate) struct CudaClient {
    /// The stream to the server.
    inner: Arc<CudaClientInner>,
}

impl CudaClient {
    /// Setup a new proving key.
    pub(crate) async fn setup(&self, elf: Elf) -> Result<CudaProvingKey, CudaClientError> {
        let request = Request::Setup { elf: elf.as_ref().into() };
        let response = self.send_and_recv(request).await?.into_result()?;

        match response {
            Response::Setup { id, vk } => Ok(CudaProvingKey::new(id, elf, vk, self.clone())),
            _ => Err(CudaClientError::UnexpectedResponse(response.type_of())),
        }
    }

    /// Create a core proof.
    pub(crate) async fn core(
        &self,
        key: &CudaProvingKey,
        stdin: SP1Stdin,
    ) -> Result<SP1CoreProof, CudaClientError> {
        let request = Request::Core { key: key.id(), stdin };
        let response = self.send_and_recv(request).await?.into_result()?;

        match response {
            Response::Core { proof } => Ok(proof),
            _ => Err(CudaClientError::UnexpectedResponse(response.type_of())),
        }
    }

    /// Compress a core proof.
    pub(crate) async fn compress(
        &self,
        vk: &SP1VerifyingKey,
        proof: SP1CoreProof,
        deferred: Vec<SP1RecursionProof<InnerSC>>,
    ) -> Result<SP1RecursionProof<InnerSC>, CudaClientError> {
        let request = Request::Compress { vk: vk.clone(), proof, deferred };
        let response = self.send_and_recv(request).await?.into_result()?;

        match response {
            Response::Compress { proof } => Ok(proof),
            _ => Err(CudaClientError::UnexpectedResponse(response.type_of())),
        }
    }

    /// Shrink a compress proof.
    pub(crate) async fn shrink(
        &self,
        proof: SP1RecursionProof<InnerSC>,
    ) -> Result<SP1RecursionProof<InnerSC>, CudaClientError> {
        let request = Request::Shrink { proof };
        let response = self.send_and_recv(request).await?.into_result()?;

        match response {
            Response::Shrink { proof } => Ok(proof),
            _ => Err(CudaClientError::UnexpectedResponse(response.type_of())),
        }
    }

    /// Wrap a shrink proof.
    pub(crate) async fn wrap(
        &self,
        proof: SP1RecursionProof<InnerSC>,
    ) -> Result<SP1RecursionProof<OuterSC>, CudaClientError> {
        let request = Request::Wrap { proof };
        let response = self.send_and_recv(request).await?.into_result()?;

        match response {
            Response::Wrap { proof } => Ok(proof),
            _ => Err(CudaClientError::UnexpectedResponse(response.type_of())),
        }
    }

    /// Remove a proving key from the server side cache.
    pub(crate) async fn destroy(&self, key: [u8; 32]) -> Result<(), CudaClientError> {
        let request = Request::Destroy { key };
        let response = self.send_and_recv(request).await?.into_result()?;

        match response {
            Response::Ok => Ok(()),
            _ => Err(CudaClientError::UnexpectedResponse(response.type_of())),
        }
    }

    async fn lock(&self) -> tokio::sync::MutexGuard<'_, UnixStream> {
        self.inner.stream.as_ref().expect("expected a valid stream").lock().await
    }
}

impl CudaClient {
    /// Connects to the server at the socket given by [`socket_path`].
    pub(crate) async fn connect(cuda_id: u32) -> Result<Self, CudaClientError> {
        CudaClientInner::connect(cuda_id).await
    }

    /// Sends a request and awaits a response, all while holding the lock on the stream.
    ///
    /// This implementation is requierd to support concurrent connections to the same device.
    pub(crate) async fn send_and_recv(
        &self,
        request: Request,
    ) -> Result<Response, CudaClientError> {
        let mut stream = self.lock().await;
        self.send(&mut stream, request).await?;
        self.recv(&mut stream).await
    }

    /// Sends a [`Request`] message to the server.
    pub(crate) async fn send(
        &self,
        stream: &mut UnixStream,
        request: Request,
    ) -> Result<(), CudaClientError> {
        self.inner.send(stream, request).await
    }

    /// Reads a [`Response`] message from the server.
    pub(crate) async fn recv(&self, stream: &mut UnixStream) -> Result<Response, CudaClientError> {
        self.inner.recv(stream).await
    }
}

struct CudaClientInner {
    stream: Option<Mutex<UnixStream>>,
}

impl CudaClientInner {
    /// Connects to the server at the socket given by [`socket_path`].
    pub(crate) async fn connect(cuda_id: u32) -> Result<CudaClient, CudaClientError> {
        // See if theres a global client still alive.
        // This may be in other instance of the client, or a proving key!
        let mut global = CLIENT.lock().await;

        // If weve already connected to this device, return that client.
        if let Some(client) = global.get(&cuda_id).and_then(|weak| weak.upgrade()) {
            tracing::debug!("Found existing client for CUDA device {}", cuda_id);
            return Ok(CudaClient { inner: client });
        }

        // Try to connect to the server, its possible this was started by another process and thats
        // why we dont have a client.
        if let Ok(client) = Self::connect_once(&socket_path(cuda_id)).await {
            tracing::debug!("Found existing server for CUDA device {}", cuda_id);

            let connection = Arc::new(client);
            let _ = global.insert(cuda_id, Arc::downgrade(&connection));
            return Ok(CudaClient { inner: connection });
        } else {
            tracing::debug!("Failed to connect to find for CUDA device {}", cuda_id);
        }

        // Print a warning if the CUDA version is not found,
        // panic if the version is too old.
        crate::check_cuda_version().await;

        // Actually start the server now that we know there isnt one running.
        crate::server::start_server(cuda_id).await?;

        // Connect to the server we just started.
        let connection = Self::connect_inner(cuda_id).await?;
        let connection = Arc::new(connection);
        let _ = global.insert(cuda_id, Arc::downgrade(&connection));

        Ok(CudaClient { inner: connection })
    }

    /// Connects to the server at [`CUDA_SOCKET`], retrying if the server is not running yet.
    async fn connect_inner(cuda_id: u32) -> Result<Self, CudaClientError> {
        let socket_path = socket_path(cuda_id);

        // Retry a few times, just in case the server hasnt started yet.
        for _ in 0..10 {
            let Ok(this) = Self::connect_once(&socket_path).await else {
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            };

            return Ok(this);
        }

        // If we get here, the server is not running yet.
        // But we want to get the actual error, so try again.
        Self::connect_once(&socket_path).await
    }

    /// Connects to the server at the given path.
    async fn connect_once(path: &Path) -> Result<Self, CudaClientError> {
        let stream = UnixStream::connect(path).await.map_err(|e| {
            CudaClientError::new_connect(e, "Could not connect to `cuslop-server` socket")
        })?;

        Ok(Self { stream: Some(Mutex::new(stream)) })
    }

    /// Sends a [`Request`] message to the server.
    pub(crate) async fn send(
        &self,
        stream: &mut UnixStream,
        request: Request,
    ) -> Result<(), CudaClientError> {
        let request_bytes = bincode::serialize(&request).map_err(CudaClientError::Serialize)?;

        let len_le = (request_bytes.len() as u32).to_le_bytes();
        stream.write_all(&len_le).await.map_err(CudaClientError::Write)?;
        stream.write_all(&request_bytes).await.map_err(CudaClientError::Write)?;

        Ok(())
    }

    /// Reads a [`Response`] message from the server.
    pub(crate) async fn recv(&self, stream: &mut UnixStream) -> Result<Response, CudaClientError> {
        // Read the length of the response.
        let mut len_le = [0; 4];
        stream.read_exact(&mut len_le).await.map_err(CudaClientError::Read)?;

        // Allocate a buffer for the response.
        let len: usize = u32::from_le_bytes(len_le) as usize;
        let mut response_bytes = vec![0; len];
        stream.read_exact(&mut response_bytes).await.map_err(CudaClientError::Read)?;

        let response =
            bincode::deserialize(&response_bytes).map_err(CudaClientError::Deserialize)?;

        Ok(response)
    }
}

/// The socket path for the given CUDA device id.
pub fn socket_path(cuda_id: u32) -> PathBuf {
    const CUDA_SOCKET_BASE: &str = "/tmp/sp1-cuda-";

    format!("{CUDA_SOCKET_BASE}{cuda_id}.sock").into()
}

impl Drop for CudaClientInner {
    fn drop(&mut self) {
        let stream = self.stream.take().expect("stream already taken");

        tokio::spawn(async move {
            let mut stream = stream.lock().await;

            if let Err(e) = stream.shutdown().await {
                tracing::error!("Failed to shutdown the stream: {}", e);
            }
        });
    }
}
