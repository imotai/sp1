// Copyright 2023 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// Allocate memory aligned to the given alignment.
///
/// Only available when the `embedded` feature is enabled.
#[allow(clippy::missing_safety_doc)]
#[no_mangle]
#[cfg(target_os = "zkvm")]
pub unsafe extern "C" fn sys_alloc_aligned(bytes: usize, align: usize) -> *mut u8 {
    use core::alloc::GlobalAlloc;
    crate::allocators::embedded::INNER_HEAP
        .alloc(std::alloc::Layout::from_size_align(bytes, align).unwrap())
}
