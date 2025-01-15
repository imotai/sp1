pub mod mpsc;

use arrayvec::ArrayVec;
use slop_baby_bear::BabyBear;

use crate::{mem::DeviceData, HostBuffer};

use super::TaskScope;

/// # Safety
///
/// TODO
pub unsafe trait CudaSend: Send {
    fn change_scope(&mut self, scope: &TaskScope);
}

unsafe impl CudaSend for () {
    fn change_scope(&mut self, _scope: &TaskScope) {}
}

/// A wrapper that forgets any scope information when sending messages.
///
/// For every type `T` that implements [Send], the type [ScopeLess<T>] will implement [CudaSend]
/// with an empty implementation of the [CudaSend::change_scope] method.
///
/// This wrapper is useful in cases in which the struct has an empty scope implementation but the
/// user has no control over the type's trait implementation.
pub struct ScopeLess<T>(pub T);

unsafe impl<T: Send> CudaSend for ScopeLess<T> {
    fn change_scope(&mut self, _scope: &TaskScope) {}
}

unsafe impl<T: CudaSend> CudaSend for Vec<T> {
    fn change_scope(&mut self, scope: &TaskScope) {
        for item in self.iter_mut() {
            item.change_scope(scope);
        }
    }
}

unsafe impl<T: DeviceData> CudaSend for HostBuffer<T> {
    fn change_scope(&mut self, _scope: &TaskScope) {}
}

unsafe impl<T: CudaSend, const N: usize> CudaSend for [T; N] {
    fn change_scope(&mut self, scope: &TaskScope) {
        for item in self.iter_mut() {
            item.change_scope(scope);
        }
    }
}

unsafe impl<T: CudaSend, const CAP: usize> CudaSend for ArrayVec<T, CAP> {
    fn change_scope(&mut self, scope: &TaskScope) {
        for item in self.iter_mut() {
            item.change_scope(scope);
        }
    }
}

macro_rules! scopeless_send_impl {
    ($($t:ty)*) => {
        $(
            unsafe impl CudaSend for $t {
                fn change_scope(&mut self, _scope: &TaskScope) {}
            }
        )*
    }
}

scopeless_send_impl!(u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64);

scopeless_send_impl!(BabyBear);

macro_rules! tuple_cuda_send_impl {
    ($(($($T:ident),+)),*) => {
        $(
            #[allow(non_snake_case)]
            unsafe impl<$($T: CudaSend),+> CudaSend for ($($T,)+) {
                fn change_scope(&mut self, scope: &TaskScope) {
                    let ($($T,)+) = self;
                    $(
                        $T.change_scope(scope);
                    )+
                }
            }
        )*
    }
}

tuple_cuda_send_impl! {
    (T1),
    (T1, T2),
    (T1, T2, T3),
    (T1, T2, T3, T4),
    (T1, T2, T3, T4, T5),
    (T1, T2, T3, T4, T5, T6),
    (T1, T2, T3, T4, T5, T6, T7),
    (T1, T2, T3, T4, T5, T6, T7, T8),
    (T1, T2, T3, T4, T5, T6, T7, T8, T9),
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10),
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11),
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12)
}

#[cfg(test)]
mod tests {
    use csl_derive::CudaSend;
    use mpsc::Channel;

    use crate::{
        cuda::{DeviceBuffer, TaskPoolBuilder},
        mem::DeviceData,
    };

    use super::*;

    #[derive(CudaSend, Debug)]
    struct MyTestStruct<T: DeviceData> {
        a: u8,
        b: u8,
        buffer: DeviceBuffer<T>,
    }

    #[tokio::test]
    async fn test_cuda_send_derive_macro() {
        let pool = TaskPoolBuilder::new().num_tasks(10).build().unwrap();
        let task_1 = pool.task().await.unwrap();
        let task_2 = pool.task().await.unwrap();
        let chan = Channel::<MyTestStruct<u8>>::new(10).unwrap();
        let (tx, mut rx) = chan.split();

        let (handle_1, handle_2) = tokio::join!(
            task_1.run(|t| async move {
                let mut buf = t.alloc::<u8>(1000000);
                buf.write_bytes(1, buf.capacity()).unwrap();
                let my_struct = MyTestStruct { a: 1, b: 2, buffer: buf };
                tx.send(&t, my_struct).await.unwrap();
            }),
            task_2.run(|t| async move {
                let msg = rx.recv(&t).await.unwrap();
                let MyTestStruct { a, b, buffer } = msg;
                assert_eq!(a, 1);
                assert_eq!(b, 2);
                assert_eq!(buffer.to_vec().await.unwrap(), vec![1; 1000000]);
            }),
        );

        handle_1.await.unwrap();
        handle_2.await.unwrap();
    }
}
