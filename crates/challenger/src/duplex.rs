use csl_cuda::TaskScope;
use slop_algebra::{Field, PrimeField64};
use slop_alloc::{Backend, Buffer, CopyIntoBackend, CpuBackend, HasBackend};
use slop_challenger::FromChallenger;
use slop_symmetric::CryptographicPermutation;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct DuplexChallenger<F, B: Backend> {
    sponge_state: Buffer<F, B>,
    input_buffer: Buffer<F, B>,
    input_buffer_size: usize,
    output_buffer: Buffer<F, B>,
    output_buffer_size: usize,
}

impl<
        F: PrimeField64,
        P: CryptographicPermutation<[F; WIDTH]>,
        const WIDTH: usize,
        const RATE: usize,
    > FromChallenger<slop_challenger::DuplexChallenger<F, P, WIDTH, RATE>, TaskScope>
    for DuplexChallenger<F, TaskScope>
where
    slop_challenger::DuplexChallenger<F, P, WIDTH, RATE>: Send + Sync,
{
    async fn from_challenger(
        challenger: &slop_challenger::DuplexChallenger<F, P, WIDTH, RATE>,
        backend: TaskScope,
    ) -> Self {
        let duplex_challenger = DuplexChallenger::from(challenger.clone());
        duplex_challenger.copy_into_backend(&backend).await.unwrap()
    }
}
impl<
        F: PrimeField64,
        P: CryptographicPermutation<[F; WIDTH]>,
        const WIDTH: usize,
        const RATE: usize,
    > From<slop_challenger::DuplexChallenger<F, P, WIDTH, RATE>>
    for DuplexChallenger<F, CpuBackend>
{
    fn from(challenger: slop_challenger::DuplexChallenger<F, P, WIDTH, RATE>) -> Self {
        let mut input_buffer = challenger.input_buffer;
        let input_buffer_size = input_buffer.len();
        assert!(input_buffer_size <= WIDTH);
        input_buffer.resize(WIDTH, F::zero());
        let mut output_buffer = challenger.output_buffer;
        let output_buffer_size = output_buffer.len();
        assert!(output_buffer_size <= WIDTH);
        output_buffer.resize(WIDTH, F::zero());
        let sponge_state = challenger.sponge_state;
        assert!(sponge_state.len() == WIDTH);

        let input_buffer = Buffer::from(input_buffer);
        let output_buffer = Buffer::from(output_buffer);
        let sponge_state = Buffer::from(sponge_state.to_vec());

        Self { input_buffer, input_buffer_size, output_buffer, output_buffer_size, sponge_state }
    }
}

impl<F, B: Backend> HasBackend for DuplexChallenger<F, B> {
    type Backend = B;

    #[inline]
    fn backend(&self) -> &Self::Backend {
        self.sponge_state.backend()
    }
}

#[repr(C)]
pub struct DuplexChallengerRawMut<F> {
    sponge_state: *mut F,
    input_buffer: *mut F,
    input_buffer_size: usize,
    output_buffer: *mut F,
    output_buffer_size: usize,
}

impl<F> DuplexChallenger<F, TaskScope> {
    pub fn as_mut_raw(&mut self) -> DuplexChallengerRawMut<F> {
        DuplexChallengerRawMut {
            sponge_state: self.sponge_state.as_mut_ptr(),
            input_buffer: self.input_buffer.as_mut_ptr(),
            input_buffer_size: self.input_buffer_size,
            output_buffer: self.output_buffer.as_mut_ptr(),
            output_buffer_size: self.output_buffer_size,
        }
    }
}

impl<F: Field> CopyIntoBackend<TaskScope, CpuBackend> for DuplexChallenger<F, CpuBackend> {
    type Output = DuplexChallenger<F, TaskScope>;

    async fn copy_into_backend(
        self,
        backend: &TaskScope,
    ) -> Result<Self::Output, slop_alloc::mem::CopyError> {
        let input_buffer = self.input_buffer.copy_into_backend(backend).await?;
        let output_buffer = self.output_buffer.copy_into_backend(backend).await?;
        let sponge_state = self.sponge_state.copy_into_backend(backend).await?;
        Ok(DuplexChallenger {
            input_buffer,
            input_buffer_size: self.input_buffer_size,
            output_buffer,
            output_buffer_size: self.output_buffer_size,
            sponge_state,
        })
    }
}

impl<F: Field> CopyIntoBackend<CpuBackend, TaskScope> for DuplexChallenger<F, TaskScope> {
    type Output = DuplexChallenger<F, CpuBackend>;

    async fn copy_into_backend(
        self,
        backend: &CpuBackend,
    ) -> Result<Self::Output, slop_alloc::mem::CopyError> {
        let input_buffer = self.input_buffer.copy_into_backend(backend).await?;
        let output_buffer = self.output_buffer.copy_into_backend(backend).await?;
        let sponge_state = self.sponge_state.copy_into_backend(backend).await?;
        Ok(DuplexChallenger {
            input_buffer,
            input_buffer_size: self.input_buffer_size,
            output_buffer,
            output_buffer_size: self.output_buffer_size,
            sponge_state,
        })
    }
}
