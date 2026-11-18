// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Callgrind client requests, so a test binary can count instructions in a window it chooses.
//!
//! Each request is valgrind's magic instruction sequence: a no-op on the CPU (the rotations add
//! up to 64 bits), which valgrind recognizes and handles. Outside valgrind it costs a few
//! instructions and does nothing. The sequences and request numbers are from valgrind's
//! `valgrind.h` and `callgrind.h`.
//!
//! `lib/llm/tests/allocations.rs` includes this file too, by `#[path]`.

#![allow(dead_code)]

use std::cell::Cell;

/// `VG_USERREQ_TOOL_BASE('C', 'T')`.
const BASE: usize = ((b'C' as usize) << 24) | ((b'T' as usize) << 16);
const TOGGLE_COLLECT: usize = BASE + 2;
const DUMP_STATS_AT: usize = BASE + 3;

/// Write the counts since the last dump to a new dump file, then zero them. `label` names the
/// dump in its `desc: Trigger:` line.
///
/// A window is the span between two dumps. Callgrind's zero request is no substitute for the
/// first one: after it, a window of N requests still carried about 4 million instructions that
/// did not grow with N (row `a`, aarch64).
pub fn dump_stats_at(label: &std::ffi::CStr) {
    request(DUMP_STATS_AT, label.as_ptr() as usize);
}

thread_local! {
    /// Whether callgrind counts the calling thread's instructions now.
    static COUNTING: Cell<bool> = const { Cell::new(true) };
}

/// Stop counting the calling thread's instructions, for the rest of its life.
pub fn stop_counting_thread() {
    std::mem::forget(pause());
}

/// Turns the calling thread's counting off until it is dropped. Pauses nest: inside one, a
/// second does nothing.
///
/// The test allocators run the system allocator in a pause. Glibc malloc's cost follows the
/// state of its heap, which follows how the threads interleave: it made a row's instructions per
/// request differ by about 1% between runs of one binary. The allocations column counts the
/// allocator's calls instead.
///
/// The toggles are explicit and per thread. Callgrind's `--toggle-collect=<function>` pairs them
/// by its call tracking, which went wrong on this code (counts that changed by 10x between
/// runs, and negative differences).
#[must_use]
pub struct Pause {
    paused: bool,
}

/// See [`Pause`]. Hold it on one thread: a guard held across an `.await` is only sound on a
/// current-thread runtime.
#[inline]
pub fn pause() -> Pause {
    // `try_with` because the allocator also runs while thread locals are torn down.
    let paused = COUNTING.try_with(|c| c.replace(false)).unwrap_or(false);
    if paused {
        request(TOGGLE_COLLECT, 0);
    }
    Pause { paused }
}

impl Drop for Pause {
    #[inline]
    fn drop(&mut self) {
        if self.paused {
            request(TOGGLE_COLLECT, 0);
            let _ = COUNTING.try_with(|c| c.set(true));
        }
    }
}

thread_local! {
    /// The pause a runtime thread holds while it is parked.
    static PARKED: Cell<Option<Pause>> = const { Cell::new(None) };
}

/// Pauses counting on the runtime's threads while they are parked. How often a thread parks and
/// wakes depends on when the other threads run: inside a request that waits for the blocking
/// thread, the parks moved row `a` by about 1,000 instructions per request between batches of
/// runs.
pub fn uncount_parked(builder: &mut tokio::runtime::Builder) -> &mut tokio::runtime::Builder {
    builder
        .on_thread_park(|| PARKED.set(Some(pause())))
        .on_thread_unpark(|| PARKED.set(None))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn request(code: usize, arg: usize) {
    let args: [usize; 6] = [code, arg, 0, 0, 0, 0];
    // SAFETY: the rotations leave x12 unchanged and `orr x10, x10, x10` is a no-op; valgrind
    // reads `args` through x4 and writes its result to x3, both declared clobbered.
    unsafe {
        std::arch::asm!(
            "ror x12, x12, #3",
            "ror x12, x12, #13",
            "ror x12, x12, #51",
            "ror x12, x12, #61",
            "orr x10, x10, x10",
            inout("x3") 0usize => _,
            in("x4") args.as_ptr(),
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn request(code: usize, arg: usize) {
    let args: [usize; 6] = [code, arg, 0, 0, 0, 0];
    // SAFETY: the rotations leave rdi unchanged and `xchg rbx, rbx` is a no-op; valgrind reads
    // `args` through rax and writes its result to rdx, declared clobbered.
    unsafe {
        std::arch::asm!(
            "rol rdi, 3",
            "rol rdi, 13",
            "rol rdi, 61",
            "rol rdi, 51",
            "xchg rbx, rbx",
            in("rax") args.as_ptr(),
            inout("rdx") 0usize => _,
        );
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn request(_code: usize, _arg: usize) {}
