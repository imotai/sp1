use csl_cuda::TaskScope;
use slop_algebra::{Field, PrimeField32};
use slop_alloc::{Backend, Buffer, CopyIntoBackend, CpuBackend, HasBackend};
use slop_challenger::FromChallenger;
use slop_symmetric::CryptographicPermutation;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct MultiField32Challenger<F, PF, B: Backend> {
    pub sponge_state: Buffer<PF, B>,
    pub input_buffer: Buffer<F, B>,
    // [input_buffer_size, output_buffer_size, num_duplex_elms, num_f_elms]
    pub buffer_sizes: Buffer<usize, B>,
    pub output_buffer: Buffer<F, B>,
}

impl<
        F: PrimeField32,
        PF: Field,
        P: CryptographicPermutation<[PF; WIDTH]>,
        const WIDTH: usize,
        const RATE: usize,
    > FromChallenger<slop_challenger::MultiField32Challenger<F, PF, P, WIDTH, RATE>, TaskScope>
    for MultiField32Challenger<F, PF, TaskScope>
where
    slop_challenger::MultiField32Challenger<F, PF, P, WIDTH, RATE>: Send + Sync,
{
    async fn from_challenger(
        challenger: &slop_challenger::MultiField32Challenger<F, PF, P, WIDTH, RATE>,
        backend: TaskScope,
    ) -> Self {
        let multi_field_challenger = MultiField32Challenger::from(challenger.clone());
        multi_field_challenger.copy_into_backend(&backend).await.unwrap()
    }
}

impl<
        F: PrimeField32,
        PF: Field,
        P: CryptographicPermutation<[PF; WIDTH]>,
        const WIDTH: usize,
        const RATE: usize,
    > From<slop_challenger::MultiField32Challenger<F, PF, P, WIDTH, RATE>>
    for MultiField32Challenger<F, PF, CpuBackend>
{
    fn from(challenger: slop_challenger::MultiField32Challenger<F, PF, P, WIDTH, RATE>) -> Self {
        let mut input_buffer = challenger.input_buffer;
        let input_buffer_size = input_buffer.len();
        assert!(input_buffer_size <= RATE * challenger.num_duplex_elms);
        input_buffer.resize(WIDTH * challenger.num_duplex_elms, F::zero());
        let mut output_buffer = challenger.output_buffer;
        let output_buffer_size = output_buffer.len();
        assert!(output_buffer_size <= WIDTH * challenger.num_f_elms);
        output_buffer.resize(WIDTH * challenger.num_f_elms, F::zero());
        let sponge_state = challenger.sponge_state;

        let input_buffer = Buffer::from(input_buffer);
        let output_buffer = Buffer::from(output_buffer);
        let sponge_state = Buffer::from(sponge_state.to_vec());
        let num_duplex_elms = challenger.num_duplex_elms;
        let num_f_elms = challenger.num_f_elms;
        let buffer_sizes =
            Buffer::from(vec![input_buffer_size, output_buffer_size, num_duplex_elms, num_f_elms]);

        Self { input_buffer, buffer_sizes, output_buffer, sponge_state }
    }
}

impl<F, PF, B: Backend> HasBackend for MultiField32Challenger<F, PF, B> {
    type Backend = B;

    #[inline]
    fn backend(&self) -> &Self::Backend {
        self.sponge_state.backend()
    }
}

#[repr(C)]
pub struct MultiField32ChallengerRawMut<F, PF> {
    pub sponge_state: *mut PF,
    pub input_buffer: *mut F,
    pub buffer_sizes: *mut usize,
    pub output_buffer: *mut F,
}

impl<F, PF> MultiField32Challenger<F, PF, TaskScope> {
    pub fn as_mut_raw(&mut self) -> MultiField32ChallengerRawMut<F, PF> {
        MultiField32ChallengerRawMut {
            sponge_state: self.sponge_state.as_mut_ptr(),
            input_buffer: self.input_buffer.as_mut_ptr(),
            buffer_sizes: self.buffer_sizes.as_mut_ptr(),
            output_buffer: self.output_buffer.as_mut_ptr(),
        }
    }
}

impl<F: PrimeField32, PF: Field> CopyIntoBackend<TaskScope, CpuBackend>
    for MultiField32Challenger<F, PF, CpuBackend>
{
    type Output = MultiField32Challenger<F, PF, TaskScope>;

    async fn copy_into_backend(
        self,
        backend: &TaskScope,
    ) -> Result<Self::Output, slop_alloc::mem::CopyError> {
        let input_buffer = self.input_buffer.copy_into_backend(backend).await?;
        let output_buffer = self.output_buffer.copy_into_backend(backend).await?;
        let sponge_state = self.sponge_state.copy_into_backend(backend).await?;
        let buffer_sizes = self.buffer_sizes.copy_into_backend(backend).await?;
        Ok(MultiField32Challenger { input_buffer, buffer_sizes, output_buffer, sponge_state })
    }
}

impl<F: Field, PF: Field> CopyIntoBackend<CpuBackend, TaskScope>
    for MultiField32Challenger<F, PF, TaskScope>
{
    type Output = MultiField32Challenger<F, PF, CpuBackend>;

    async fn copy_into_backend(
        self,
        backend: &CpuBackend,
    ) -> Result<Self::Output, slop_alloc::mem::CopyError> {
        let input_buffer = self.input_buffer.copy_into_backend(backend).await?;
        let output_buffer = self.output_buffer.copy_into_backend(backend).await?;
        let sponge_state = self.sponge_state.copy_into_backend(backend).await?;
        let buffer_sizes = self.buffer_sizes.copy_into_backend(backend).await?;
        Ok(MultiField32Challenger { input_buffer, buffer_sizes, output_buffer, sponge_state })
    }
}
