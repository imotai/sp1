use csl_cuda::TaskScope;
use csl_prover::{local_gpu_opts, CudaSP1ProverComponents, SP1CudaProverBuilder};
use sp1_core_executor::{Program, SP1Context, SP1RecursionProof};
use sp1_core_machine::io::SP1Stdin;
use sp1_cuda::{
    api::{Request, Response},
    client::socket_path,
};
use sp1_primitives::SP1GlobalContext;
use sp1_prover::{local::LocalProver, InnerSC, SP1CoreProof, SP1VerifyingKey};
use std::collections::HashMap;

use std::io;
use std::sync::Arc;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{UnixListener, UnixStream},
};

type Prover = LocalProver<CudaSP1ProverComponents>;

/// A cached proving key and verifying key.
#[derive(Clone)]
struct CachedProgram {
    program: Arc<Program>,
    vk: SP1VerifyingKey,
}

/// The server for the cuslop service.
pub struct Server {
    pub cuda_device_id: u32,
}

/// The context for a single connection to the server.
struct ConnectionCtx {
    pk_cache: HashMap<[u8; 32], CachedProgram>,
    prover: Arc<Prover>,
}

impl Server {
    /// Run the server, indefinitely.
    pub async fn run(self, task_scope: TaskScope) {
        let socket_path = socket_path(self.cuda_device_id);

        // Try to remove the socket file socket incase the file was never cleaned up.
        if let Err(e) = std::fs::remove_file(&socket_path) {
            tracing::warn!("Failed to remove orphaned socket: {}", e);
        }

        let listener = UnixListener::bind(&socket_path).expect("Failed to bind to socket addr");

        let prover = SP1CudaProverBuilder::new(task_scope).without_vk_verification().build().await;
        let prover = LocalProver::new(prover, local_gpu_opts());
        let prover = Arc::new(prover);

        tracing::info!("Server listening @ {}", socket_path.display());
        loop {
            tokio::select! {
                res = listener.accept() => {
                    if let Ok((stream, _)) = res {
                        tracing::info!("Connection accepted");

                        let prover = prover.clone();

                        tokio::spawn(async move {
                            let mut stream = stream;

                            if let Err(e) = Self::handle_connection(prover, &mut stream).await {
                                if e.kind() == io::ErrorKind::UnexpectedEof
                                    || e.kind() == io::ErrorKind::BrokenPipe
                                {
                                    tracing::info!("Connection disconnected");
                                    let _ = send_response(&mut stream, Response::ConnectionClosed).await;
                                } else {
                                    tracing::error!("Error handling connection: {:?}", e);
                                }
                            }
                        });
                    }
                }
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("Ctrl-C received, shutting down");

                    // Remove the socket file, explicitly.
                    if let Err(e) = std::fs::remove_file(&socket_path) {
                        tracing::error!("Failed to remove orphaned socket: {}", e);
                    }

                    break;
                }
            }
        }
    }

    async fn handle_connection(
        prover: Arc<Prover>,
        stream: &mut UnixStream,
    ) -> Result<(), io::Error> {
        let mut ctx = ConnectionCtx { pk_cache: Default::default(), prover };

        loop {
            let mut len = [0_u8; 4];
            stream.read_exact(&mut len).await?;

            let len = u32::from_le_bytes(len);
            let mut request_buf = vec![0; len as usize];
            stream.read_exact(&mut request_buf).await?;

            let request: Request = match bincode::deserialize(&request_buf) {
                Ok(request) => request,
                Err(e) => {
                    eprintln!("Error deserializing request: {e}");
                    let response = Response::InternalError(e.to_string());
                    send_response(stream, response).await?;
                    return Ok(());
                }
            };

            let response = Self::handle_request(&mut ctx, request).await;
            send_response(stream, response).await?;
        }
    }

    async fn handle_request(ctx: &mut ConnectionCtx, request: Request) -> Response {
        match request {
            Request::Setup { elf } => ctx.setup(&elf).await,
            Request::Core { key: id, stdin, proof_nonce } => {
                ctx.prove_core(id, stdin, proof_nonce).await
            }
            Request::Compress { vk, proof, deferred } => {
                ctx.prove_compress(vk, proof, deferred).await
            }
            Request::Shrink { proof } => ctx.prove_shrink(proof).await,
            Request::Wrap { proof } => ctx.prove_wrap(proof).await,
            Request::Destroy { key } => ctx.destroy(key).await,
        }
    }
}

impl ConnectionCtx {
    pub async fn destroy(&mut self, key: [u8; 32]) -> Response {
        tracing::info!("Destroying key");

        self.pk_cache.remove(&key);

        Response::Ok
    }

    pub async fn setup(&mut self, elf: &[u8]) -> Response {
        tracing::info!("Running setup");

        let elf_hash = hash_elf(elf);

        // Happy path if we already know the PK.
        if let Some(pk) = self.pk_cache.get(&elf_hash) {
            return Response::Setup { id: elf_hash, vk: pk.vk.clone() };
        }

        // Get the proving key.
        let (_, program, vk) = self.prover.prover().core().setup(elf).await;

        // Insert the pk into the cache.
        self.pk_cache.insert(elf_hash, CachedProgram { program, vk: vk.clone() });

        Response::Setup { id: elf_hash, vk }
    }

    pub async fn prove_core(
        &mut self,
        id: [u8; 32],
        stdin: SP1Stdin,
        proof_nonce: [u32; 4],
    ) -> Response {
        tracing::info!("Proving core");

        let Some(cached) = self.pk_cache.get(&id) else {
            return Response::InternalError(
                "Missing proving key for core proof, do not drop the prover while maintaing a proving key generated by it.".to_string(),
            );
        };

        let pk = self
            .prover
            .prover()
            .core()
            .setup_with_vk(cached.program.clone(), cached.vk.clone())
            .await;
        let pk = unsafe { pk.into_inner() };

        let context = SP1Context::builder().proof_nonce(proof_nonce).build();

        match self.prover.clone().prove_core(pk, cached.program.clone(), stdin, context).await {
            Ok(proof) => Response::Core { proof },
            Err(e) => Response::ProverError(e.to_string()),
        }
    }

    pub async fn prove_compress(
        &mut self,
        vk: SP1VerifyingKey,
        proof: SP1CoreProof,
        deferred_proofs: Vec<SP1RecursionProof<SP1GlobalContext, InnerSC>>,
    ) -> Response {
        tracing::info!("Proving compress");

        match self.prover.clone().compress(&vk, proof, deferred_proofs).await {
            Ok(proof) => Response::Compress { proof },
            Err(e) => Response::ProverError(e.to_string()),
        }
    }

    pub async fn prove_shrink(
        &mut self,
        proof: SP1RecursionProof<SP1GlobalContext, InnerSC>,
    ) -> Response {
        tracing::info!("Proving shrink");

        match self.prover.shrink(proof).await {
            Ok(proof) => Response::Shrink { proof },
            Err(e) => Response::ProverError(e.to_string()),
        }
    }

    pub async fn prove_wrap(
        &mut self,
        proof: SP1RecursionProof<SP1GlobalContext, InnerSC>,
    ) -> Response {
        tracing::info!("Proving wrap");

        match self.prover.wrap(proof).await {
            Ok(proof) => Response::Wrap { proof },
            Err(e) => Response::ProverError(e.to_string()),
        }
    }
}

fn hash_elf(elf: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(elf);
    hasher.finalize().into()
}

async fn send_response(stream: &mut UnixStream, response: Response) -> Result<(), io::Error> {
    let response_bytes = bincode::serialize(&response).unwrap();
    let len = response_bytes.len() as u32;
    stream.write_all(&len.to_le_bytes()).await?;
    stream.write_all(&response_bytes).await?;

    Ok(())
}
