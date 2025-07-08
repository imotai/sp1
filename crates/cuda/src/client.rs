use sp1_core_machine::{io::SP1Stdin, recursion::SP1RecursionProof};
use sp1_primitives::Elf;
use sp1_prover::{InnerSC, OuterSC, SP1CoreProof, SP1VerifyingKey};
use std::{
    collections::HashMap,
    io::Error as IoError,
    path::PathBuf,
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
    MIN_CUDA_VERSION,
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

#[derive(Debug, thiserror::Error)]
pub enum CudaClientError {
    #[error("Failed to connect to the server: {0:?}")]
    Connect(IoError),

    #[error("Failed to serialize the request: {0:?}")]
    Serialize(bincode::Error),

    #[error("Failed to deserialize the response: {0:?}")]
    Deserialize(bincode::Error),

    #[error("Failed to write the request: {0:?}")]
    Write(IoError),

    #[error("Failed to read the response: {0:?}")]
    Read(IoError),

    #[error("The server returned an internal error \n {0}")]
    ServerError(String),

    #[error("The server returned an unexpected response: {0:?}")]
    UnexpectedResponse(&'static str),

    #[error("The server returned a prover error: {0:?}")]
    ProverError(String),

    #[error("Failed to download the server: {0:?}")]
    Download(#[from] reqwest::Error),

    #[error("Failed to download the server: {0:?}")]
    DownloadIO(std::io::Error),

    #[error("CUDA version is too old. Please upgrade to at least {}", MIN_CUDA_VERSION)]
    CudaVersionTooOld,

    #[error("Unexpected error: {0:?}")]
    Unexpected(String),
}

impl CudaClient {
    /// Setup a new proving key.
    pub(crate) async fn setup(&self, elf: Elf) -> Result<CudaProvingKey, CudaClientError> {
        let request = Request::Setup { elf: elf.as_ref().into() };
        self.send(request).await?;

        let response = self.recv().await?.into_result()?;

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
        // Send the core request.
        let request = Request::Core { key: key.id(), stdin };
        self.send(request).await?;

        // Receive the response.
        let response = self.recv().await?.into_result()?;

        // Return the proof.
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
        self.send(request).await?;

        let response = self.recv().await?.into_result()?;

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
        // Send the shrink request.
        let request = Request::Shrink { proof };
        self.send(request).await?;

        // Receive the response.
        let response = self.recv().await?.into_result()?;

        // Return the proof.
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
        self.send(request).await?;

        let response = self.recv().await?.into_result()?;

        match response {
            Response::Wrap { proof } => Ok(proof),
            _ => Err(CudaClientError::UnexpectedResponse(response.type_of())),
        }
    }

    /// Remove a proving key from the server side cache.
    pub(crate) async fn destroy(&self, key: [u8; 32]) -> Result<(), CudaClientError> {
        let request = Request::Destroy { key };
        self.send(request).await?;

        let response = self.recv().await?.into_result()?;

        match response {
            Response::Ok => Ok(()),
            Response::InternalError(e) => Err(CudaClientError::ServerError(e)),
            _ => Err(CudaClientError::UnexpectedResponse(response.type_of())),
        }
    }
}

impl CudaClient {
    /// Connects to the server at the socket given by [`socket_path`].
    pub(crate) async fn connect(cuda_id: u32) -> Result<Self, CudaClientError> {
        CudaClientInner::connect(cuda_id).await
    }

    /// Sends a [`Request`] message to the server.
    pub(crate) async fn send(&self, request: Request) -> Result<(), CudaClientError> {
        self.inner.send(request).await
    }

    /// Reads a [`Response`] message from the server.
    pub(crate) async fn recv(&self) -> Result<Response, CudaClientError> {
        self.inner.recv().await
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

        // If there is, return that client.
        if let Some(client) = global.get(&cuda_id).and_then(|weak| weak.upgrade()) {
            tracing::debug!("Found existing client for CUDA device {}", cuda_id);
            return Ok(CudaClient { inner: client });
        }

        crate::check_cuda_version().await?;

        // There is no clients for this CUDA device, we need to start the server.
        crate::server::start_server(cuda_id).await?;

        // Connect to the server we just started.
        let connection = Self::connect_once(cuda_id).await?;
        let connection = Arc::new(connection);
        let _ = global.insert(cuda_id, Arc::downgrade(&connection));

        Ok(CudaClient { inner: connection })
    }

    /// Connects to the server at [`CUDA_SOCKET`], without checking for a global client.
    async fn connect_once(cuda_id: u32) -> Result<Self, CudaClientError> {
        let socket_path = socket_path(cuda_id);

        // Retry a few times, wait for the server to start.
        for _ in 0..10 {
            let Ok(stream) = UnixStream::connect(&socket_path).await else {
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            };

            return Ok(Self { stream: Some(Mutex::new(stream)) });
        }

        // If we get here, the server is not running yet.
        // But we want to get the actual error, so try again.
        let stream = UnixStream::connect(socket_path).await.map_err(CudaClientError::Connect)?;

        Ok(Self { stream: Some(Mutex::new(stream)) })
    }

    /// Sends a [`Request`] message to the server.
    pub(crate) async fn send(&self, request: Request) -> Result<(), CudaClientError> {
        let mut stream = self.stream.as_ref().expect("expected a valid stream").lock().await;
        let request_bytes = bincode::serialize(&request).map_err(CudaClientError::Serialize)?;

        let len_le = (request_bytes.len() as u32).to_le_bytes();
        stream.write_all(&len_le).await.map_err(CudaClientError::Write)?;
        stream.write_all(&request_bytes).await.map_err(CudaClientError::Write)?;

        Ok(())
    }

    /// Reads a [`Response`] message from the server.
    pub(crate) async fn recv(&self) -> Result<Response, CudaClientError> {
        let mut stream = self.stream.as_ref().expect("expected a valid stream").lock().await;

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
