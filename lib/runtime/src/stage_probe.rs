// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scratch per-request stage timestamps for the response plane cost study.
//!
//! Off unless `DYN_STAGE_PROBE_DIR` is set. When on, each call takes one
//! `CLOCK_MONOTONIC` read and pushes one record under a mutex; a background
//! thread appends the records to `<dir>/probe_<pid>.csv` once a second. The
//! clock is system-wide, so records from the frontend and the workers on one
//! host can be joined by request id. Calls happen a few times per request,
//! never per token.

use parking_lot::Mutex;
use std::{io::Write, sync::OnceLock, time::Duration};

pub const F_START: &str = "f_start";
pub const F_REGISTERED: &str = "f_registered";
pub const F_DISPATCHED: &str = "f_dispatched";
pub const F_PROLOGUE: &str = "f_prologue";
pub const F_FIRST_ITEM: &str = "f_first_item";
pub const W_START: &str = "w_start";
pub const W_PEER_READY: &str = "w_peer_ready";
pub const W_OPENED: &str = "w_opened";
pub const W_PROLOGUE_SENT: &str = "w_prologue_sent";
pub const W_FIRST_FROM_ENGINE: &str = "w_first_from_engine";
pub const W_FIRST_SENT: &str = "w_first_sent";

struct Sink {
    records: Mutex<Vec<(&'static str, String, u64)>>,
}

static SINK: OnceLock<Option<&'static Sink>> = OnceLock::new();

fn sink() -> Option<&'static Sink> {
    *SINK.get_or_init(|| {
        let dir = std::env::var_os("DYN_STAGE_PROBE_DIR")?;
        let path = std::path::Path::new(&dir).join(format!("probe_{}.csv", std::process::id()));
        let sink: &'static Sink = Box::leak(Box::new(Sink {
            records: Mutex::new(Vec::with_capacity(1 << 16)),
        }));
        std::thread::Builder::new()
            .name("stage-probe".into())
            .spawn(move || {
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .expect("open stage probe file");
                let mut batch = Vec::with_capacity(1 << 16);
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                    std::mem::swap(&mut *sink.records.lock(), &mut batch);
                    let mut out = String::with_capacity(batch.len() * 64);
                    for (stage, id, ns) in batch.drain(..) {
                        out.push_str(stage);
                        out.push(',');
                        out.push_str(&id);
                        out.push(',');
                        out.push_str(&ns.to_string());
                        out.push('\n');
                    }
                    if !out.is_empty() {
                        let _ = file.write_all(out.as_bytes());
                        let _ = file.flush();
                    }
                }
            })
            .expect("spawn stage probe writer");
        Some(sink)
    })
}

fn monotonic_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable timespec.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// Record that request `id` reached `stage` now.
#[inline]
pub fn mark(stage: &'static str, id: &str) {
    if let Some(sink) = sink() {
        let ns = monotonic_ns();
        sink.records.lock().push((stage, id.to_owned(), ns));
    }
}
