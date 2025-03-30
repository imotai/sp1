mod memory;
mod program;
mod word;

pub use memory::*;
pub use program::*;
pub use word::*;

use sp1_stark::air::{BaseAirBuilder, SP1AirBuilder};

use crate::{memory::MemoryAccessColsU8, operations::U16toU8Operation};

/// A trait which contains methods related to memory interactions in an AIR.
pub trait SP1CoreAirBuilder:
    SP1AirBuilder + WordAirBuilder + MemoryAirBuilder + ProgramAirBuilder
{
    fn generate_limbs(
        &mut self,
        memory_access_cols: &[MemoryAccessColsU8<Self::Var>],
        is_real: Self::Expr,
    ) -> Vec<Self::Expr> {
        // Convert the u16 limbs to u8 limbs using the safe API with range checks.
        let limbs = memory_access_cols
            .iter()
            .flat_map(|access| {
                U16toU8Operation::<Self::F>::eval_u16_to_u8_safe(
                    self,
                    access.memory_access.prev_value.0.map(|x| x.into()),
                    access.prev_value_u8,
                    is_real.clone(),
                )
            })
            .collect::<Vec<_>>();
        limbs
    }
}

impl<AB: BaseAirBuilder> MemoryAirBuilder for AB {}
impl<AB: BaseAirBuilder> ProgramAirBuilder for AB {}
impl<AB: BaseAirBuilder> WordAirBuilder for AB {}
impl<AB: BaseAirBuilder + SP1AirBuilder> SP1CoreAirBuilder for AB {}
