use csl_cuda::TaskScope;
use slop_algebra::Field;
use slop_alloc::{Backend, Buffer, CopyToBackend, CpuBackend, HasBackend};

pub struct MerkleTreeHasher<F: Field, A: Backend, const WIDTH: usize> {
    pub internal_constants: Buffer<F, A>,
    pub external_constants: Buffer<[F; WIDTH], A>,
    pub diffusion_matrix: Buffer<F, A>,
    pub monty_inverse: F,
}

#[repr(C)]
pub struct MerkleTreeHasherRaw<F: Field> {
    pub internal_constants: *const F,
    pub external_constants: *const F,
    pub diffusion_matrix: *const F,
    pub monty_inverse: F,
}

impl<F: Field, A: Backend, const WIDTH: usize> MerkleTreeHasher<F, A, WIDTH> {
    pub fn new(
        internal_constants: Buffer<F, A>,
        external_constants: Buffer<[F; WIDTH], A>,
        diffusion_matrix: Buffer<F, A>,
        monty_inverse: F,
    ) -> Self {
        Self { internal_constants, external_constants, diffusion_matrix, monty_inverse }
    }

    pub fn as_raw(&self) -> MerkleTreeHasherRaw<F> {
        MerkleTreeHasherRaw {
            internal_constants: self.internal_constants.as_ptr(),
            external_constants: self.external_constants.as_ptr() as *const F,
            diffusion_matrix: self.diffusion_matrix.as_ptr(),
            monty_inverse: self.monty_inverse,
        }
    }
}

/// implement default for merkletreehasher
impl<F: Field, const WIDTH: usize> Default for MerkleTreeHasher<F, CpuBackend, WIDTH> {
    fn default() -> Self {
        Self {
            internal_constants: Buffer::default(),
            external_constants: Buffer::default(),
            diffusion_matrix: Buffer::default(),
            monty_inverse: F::one(),
        }
    }
}

impl<F: Field, const WIDTH: usize, A: Backend> HasBackend for MerkleTreeHasher<F, A, WIDTH> {
    type Backend = A;

    #[inline]
    fn backend(&self) -> &Self::Backend {
        self.internal_constants.backend()
    }
}

impl<F: Field, const WIDTH: usize> CopyToBackend<TaskScope, CpuBackend>
    for MerkleTreeHasher<F, CpuBackend, WIDTH>
{
    type Output = MerkleTreeHasher<F, TaskScope, WIDTH>;

    async fn copy_to_backend(
        &self,
        backend: &TaskScope,
    ) -> Result<Self::Output, slop_alloc::mem::CopyError> {
        let internal_constants = self.internal_constants.copy_to_backend(backend).await?;
        let external_constants = self.external_constants.copy_to_backend(backend).await?;
        let diffusion_matrix = self.diffusion_matrix.copy_to_backend(backend).await?;
        Ok(MerkleTreeHasher {
            internal_constants,
            external_constants,
            diffusion_matrix,
            monty_inverse: self.monty_inverse,
        })
    }
}

impl<F: Field, const WIDTH: usize> CopyToBackend<CpuBackend, TaskScope>
    for MerkleTreeHasher<F, TaskScope, WIDTH>
{
    type Output = MerkleTreeHasher<F, CpuBackend, WIDTH>;

    async fn copy_to_backend(
        &self,
        backend: &CpuBackend,
    ) -> Result<Self::Output, slop_alloc::mem::CopyError> {
        let internal_constants = self.internal_constants.copy_to_backend(backend).await?;
        let external_constants = self.external_constants.copy_to_backend(backend).await?;
        let diffusion_matrix = self.diffusion_matrix.copy_to_backend(backend).await?;
        Ok(MerkleTreeHasher {
            internal_constants,
            external_constants,
            diffusion_matrix,
            monty_inverse: self.monty_inverse,
        })
    }
}
