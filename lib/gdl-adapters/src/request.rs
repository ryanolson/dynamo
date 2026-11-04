// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a gdl pipeline carries a Dynamo request.

use std::mem::MaybeUninit;
use std::sync::{Mutex, PoisonError};

use dynamo_runtime::engine::Data;
use dynamo_runtime::pipeline::SingleIn;

/// A Dynamo request on the heap, as a gdl pipeline carries a large one.
///
/// gdl moves a request by value through every node of a hop: into the chain's futures, into
/// each node and out of it. A chat request is about 2 KB and a preprocessed request about
/// 1 KB, so each move is a `memcpy`. Boxed, each move is a pointer, for one allocation where
/// the request enters the pipeline or a node builds it.
pub type Boxed<T> = Box<SingleIn<T>>;

/// How a gdl pipeline carries a Dynamo request of type `T`: as `SingleIn<T>` itself, for a
/// small request, or as [`Boxed<T>`].
pub trait Carried<T: Data>: Sized + Send + 'static {
    /// The memory a request leaves when [`take`](Self::take) moves it out of its carrier, kept
    /// for the next request a node sends: none for `SingleIn<T>`, the box for [`Boxed<T>`].
    type Spare: Default + Send + Sync + 'static;

    /// `request` in a carrier of its own.
    fn carry(request: SingleIn<T>) -> Self;

    /// `request` in the memory `spare` holds, or in a carrier of its own once `spare` is used.
    fn carry_in(request: SingleIn<T>, spare: &Self::Spare) -> Self;

    /// The request, out of its carrier.
    fn into_request(self) -> SingleIn<T>;

    /// The request, out of its carrier, and the memory it leaves for
    /// [`carry_in`](Self::carry_in).
    fn take(self) -> (SingleIn<T>, Self::Spare);
}

impl<T: Data> Carried<T> for SingleIn<T> {
    type Spare = ();

    #[inline]
    fn carry(request: SingleIn<T>) -> Self {
        request
    }

    #[inline]
    fn carry_in(request: SingleIn<T>, _spare: &()) -> Self {
        request
    }

    #[inline]
    fn into_request(self) -> SingleIn<T> {
        self
    }

    #[inline]
    fn take(self) -> (SingleIn<T>, ()) {
        (self, ())
    }
}

impl<T: Data> Carried<T> for Boxed<T> {
    type Spare = SpareBox<T>;

    #[inline]
    fn carry(request: SingleIn<T>) -> Self {
        Box::new(request)
    }

    #[inline]
    fn carry_in(request: SingleIn<T>, spare: &SpareBox<T>) -> Self {
        match spare.take() {
            Some(memory) => Box::write(memory, request),
            None => Box::new(request),
        }
    }

    #[inline]
    fn into_request(self) -> SingleIn<T> {
        *self
    }

    #[inline]
    fn take(self) -> (SingleIn<T>, SpareBox<T>) {
        let (request, memory) = take(self);
        (request, SpareBox(Mutex::new(Some(memory))))
    }
}

/// The box a [`Boxed<T>`] request left, for one more request of the same type.
///
/// A node that takes a request apart and sends a new one on, as Migration does for each
/// attempt, sends its first in the box it was given, so the node adds no allocation. A `Mutex`,
/// as the node's body sends through a shared reference.
pub struct SpareBox<T: Data>(Mutex<Option<Box<MaybeUninit<SingleIn<T>>>>>);

impl<T: Data> SpareBox<T> {
    fn take(&self) -> Option<Box<MaybeUninit<SingleIn<T>>>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).take()
    }
}

impl<T: Data> Default for SpareBox<T> {
    fn default() -> Self {
        Self(Mutex::new(None))
    }
}

/// The value out of `boxed`, and its memory, to hold another value of the same type.
///
/// `std`'s `Box::take` does this, but is not stable. Dropped unused, the memory is freed and
/// nothing in it is dropped.
fn take<T>(boxed: Box<T>) -> (T, Box<MaybeUninit<T>>) {
    let raw = Box::into_raw(boxed);
    // SAFETY: `raw` comes from `Box::into_raw`, so it is aligned, initialized, and owned here
    // alone. The read moves the value out once. The allocation is then owned again as
    // `MaybeUninit<T>`, which has the layout of `T` and never drops its contents, so the value
    // that was moved out is not dropped twice.
    unsafe { (raw.read(), Box::from_raw(raw.cast::<MaybeUninit<T>>())) }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use dynamo_runtime::pipeline::Context;

    use super::*;

    /// Counts its drops.
    struct Counted(Arc<AtomicUsize>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn counted() -> (Arc<AtomicUsize>, SingleIn<Counted>) {
        let drops = Arc::new(AtomicUsize::new(0));
        (drops.clone(), Context::new(Counted(drops)))
    }

    #[test]
    fn the_spare_box_holds_the_next_request() {
        let (first_drops, first) = counted();
        let boxed = <Boxed<Counted>>::carry(first);
        let address = std::ptr::from_ref::<SingleIn<Counted>>(&boxed).addr();
        let (first, spare) = boxed.take();
        assert_eq!(
            first_drops.load(Ordering::SeqCst),
            0,
            "taken out, not dropped"
        );

        let (second_drops, second) = counted();
        let second = <Boxed<Counted>>::carry_in(second, &spare);
        assert_eq!(
            std::ptr::from_ref::<SingleIn<Counted>>(&second).addr(),
            address,
            "the second request is in the first one's box"
        );
        // Used once: the next request gets a box of its own.
        let (third_drops, third) = counted();
        let third = <Boxed<Counted>>::carry_in(third, &spare);
        assert_ne!(
            std::ptr::from_ref::<SingleIn<Counted>>(&third).addr(),
            address
        );

        drop(first);
        drop(second);
        drop(third);
        for drops in [first_drops, second_drops, third_drops] {
            assert_eq!(drops.load(Ordering::SeqCst), 1, "each request dropped once");
        }
    }

    #[test]
    fn an_unused_spare_box_drops_nothing() {
        let (drops, request) = counted();
        let (request, spare) = <Boxed<Counted>>::carry(request).take();
        drop(spare);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "the spare holds no request"
        );
        drop(request);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn an_unboxed_request_is_carried_as_it_is() {
        let (drops, request) = counted();
        let id = request.id().to_string();
        let (request, ()) = <SingleIn<Counted>>::carry_in(request, &()).take();
        assert_eq!(request.id(), id);
        drop(request);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
