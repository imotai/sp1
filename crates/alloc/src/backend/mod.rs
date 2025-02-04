mod cpu;

use std::{borrow::Cow, rc::Rc, sync::Arc};

pub use cpu::*;

use crate::{mem::DeviceMemory, Allocator};

/// # Safety
///
/// TODO
pub unsafe trait Backend: Sized + Allocator + DeviceMemory + Clone {}

pub trait HasBackend {
    type Backend: Backend;

    fn backend(&self) -> &Self::Backend;
}

impl<'a, T> HasBackend for &'a T
where
    T: HasBackend,
{
    type Backend = T::Backend;

    fn backend(&self) -> &Self::Backend {
        (**self).backend()
    }
}

impl<'a, T> HasBackend for &'a mut T
where
    T: HasBackend,
{
    type Backend = T::Backend;

    fn backend(&self) -> &Self::Backend {
        (**self).backend()
    }
}

impl<'a, T> HasBackend for Cow<'a, T>
where
    T: HasBackend + Clone,
{
    type Backend = T::Backend;

    fn backend(&self) -> &Self::Backend {
        self.as_ref().backend()
    }
}

impl<T> HasBackend for Box<T>
where
    T: HasBackend,
{
    type Backend = T::Backend;

    fn backend(&self) -> &Self::Backend {
        self.as_ref().backend()
    }
}

impl<T> HasBackend for Arc<T>
where
    T: HasBackend,
{
    type Backend = T::Backend;

    fn backend(&self) -> &Self::Backend {
        self.as_ref().backend()
    }
}

impl<T> HasBackend for Rc<T>
where
    T: HasBackend,
{
    type Backend = T::Backend;

    fn backend(&self) -> &Self::Backend {
        self.as_ref().backend()
    }
}
