//! Regression tests for custom-controller callback ownership.
//!
//! The allocation passed to `MaaCustomControllerCreate` is referenced by native code
//! until `MaaControllerDestroy` returns. These tests pin down when it is released:
//! after the last owner drops, after native destroy has joined its action thread, and
//! exactly once for clones, record wrappers, bound taskers, concurrent drops and
//! failure paths.

#![cfg(feature = "custom")]

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex, mpsc};
use std::thread;
use std::time::Duration;

use maa_framework::{
    controller::Controller, custom_controller::CustomControllerCallback, tasker::Tasker,
};

#[derive(Default)]
struct Counters {
    drops: AtomicUsize,
    connects: AtomicUsize,
    connected_calls: AtomicUsize,
}

/// Counts releases of the callback allocation.
struct DropProbe(Arc<Counters>);

impl CustomControllerCallback for DropProbe {
    fn connect(&self) -> bool {
        self.0.connects.fetch_add(1, Ordering::SeqCst);
        true
    }

    fn connected(&self) -> bool {
        self.0.connected_calls.fetch_add(1, Ordering::SeqCst);
        true
    }
}

impl Drop for DropProbe {
    fn drop(&mut self) {
        self.0.drops.fetch_add(1, Ordering::SeqCst);
    }
}

/// Always fails to connect, to exercise the failed-connect path.
struct FailingProbe(Arc<Counters>);

impl CustomControllerCallback for FailingProbe {
    fn connect(&self) -> bool {
        self.0.connects.fetch_add(1, Ordering::SeqCst);
        false
    }
}

impl Drop for FailingProbe {
    fn drop(&mut self) {
        self.0.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Default)]
struct BlockingState {
    entered: Mutex<bool>,
    release: Mutex<bool>,
    wake: Condvar,
    connects: AtomicUsize,
    drops: AtomicUsize,
    callback_running: AtomicBool,
    callback_returned: AtomicBool,
    freed_while_running: AtomicBool,
}

/// Blocks inside `connect` until the test releases it.
struct BlockingProbe(Arc<BlockingState>);

impl CustomControllerCallback for BlockingProbe {
    fn connect(&self) -> bool {
        let state = &self.0;
        state.connects.fetch_add(1, Ordering::SeqCst);
        state.callback_running.store(true, Ordering::SeqCst);

        *state.entered.lock().unwrap() = true;
        state.wake.notify_all();

        let mut release = state.release.lock().unwrap();
        while !*release {
            release = state.wake.wait(release).unwrap();
        }
        drop(release);

        state.callback_running.store(false, Ordering::SeqCst);
        state.callback_returned.store(true, Ordering::SeqCst);
        true
    }
}

impl Drop for BlockingProbe {
    fn drop(&mut self) {
        if self.0.callback_running.load(Ordering::SeqCst) {
            self.0.freed_while_running.store(true, Ordering::SeqCst);
        }
        self.0.drops.fetch_add(1, Ordering::SeqCst);
    }
}

fn temp_recording_path(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "maa-custom-controller-{}-{}",
        std::process::id(),
        name
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn custom_controller_callback_lifetime() {
    let counters = Arc::new(Counters::default());
    let controller =
        Controller::new_custom(DropProbe(counters.clone())).expect("new_custom should succeed");
    let clone = controller.clone();
    drop(controller);
    assert_eq!(
        counters.drops.load(Ordering::SeqCst),
        0,
        "callback dropped while a clone is still alive"
    );
    drop(clone);
    assert_eq!(
        counters.drops.load(Ordering::SeqCst),
        1,
        "callback not released after the last clone"
    );

    for _ in 0..1000 {
        drop(
            Controller::new_custom(DropProbe(counters.clone())).expect("new_custom should succeed"),
        );
    }
    assert_eq!(counters.drops.load(Ordering::SeqCst), 1001);
    assert_eq!(Arc::strong_count(&counters), 1);
}

#[test]
fn destroy_waits_for_in_flight_callback() {
    let state = Arc::new(BlockingState::default());
    let controller =
        Controller::new_custom(BlockingProbe(state.clone())).expect("new_custom should succeed");

    let connect_id = controller
        .post_connection()
        .expect("post_connection should succeed");

    let mut entered = state.entered.lock().unwrap();
    while !*entered {
        let (guard, timeout) = state
            .wake
            .wait_timeout(entered, Duration::from_secs(5))
            .unwrap();
        entered = guard;
        assert!(
            !timeout.timed_out(),
            "connect callback was never invoked for id {connect_id}"
        );
    }
    drop(entered);

    let (started_tx, started_rx) = mpsc::channel();
    let (dropped_tx, dropped_rx) = mpsc::channel();
    let dropper = thread::spawn(move || {
        let _ = started_tx.send(());
        drop(controller);
        let _ = dropped_tx.send(());
    });
    started_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("drop thread did not start");

    // The callback is pinned by the condvar, so it cannot complete until this test
    // releases it. The bounded wait therefore detects an early MaaControllerDestroy
    // return rather than scheduling noise.
    assert!(
        dropped_rx.recv_timeout(Duration::from_millis(500)).is_err(),
        "MaaControllerDestroy returned while a native callback was still running"
    );
    assert_eq!(
        state.drops.load(Ordering::SeqCst),
        0,
        "callback freed while native still uses it"
    );

    *state.release.lock().unwrap() = true;
    state.wake.notify_all();

    dropped_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("destroy did not finish after the callback returned");
    dropper.join().unwrap();

    assert_eq!(state.connects.load(Ordering::SeqCst), 1);
    assert!(state.callback_returned.load(Ordering::SeqCst));
    assert!(
        !state.freed_while_running.load(Ordering::SeqCst),
        "callback allocation was freed while connect was still running"
    );
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
}

#[test]
fn record_controller_retains_custom_callback() {
    let dir = temp_recording_path("record");
    let recording = dir.join("recording.jsonl");
    let counters = Arc::new(Counters::default());

    let controller =
        Controller::new_custom(DropProbe(counters.clone())).expect("new_custom should succeed");
    let record = Controller::new_record(&controller, recording.to_str().unwrap())
        .expect("new_record should succeed");
    drop(controller);

    assert_eq!(
        counters.drops.load(Ordering::SeqCst),
        0,
        "record wrapper released the inner callback too early"
    );
    assert!(
        record.connected(),
        "record wrapper must keep the inner control unit usable"
    );
    assert!(
        counters.connected_calls.load(Ordering::SeqCst) >= 1,
        "record wrapper did not reach the inner connected callback"
    );

    drop(record);
    assert_eq!(counters.drops.load(Ordering::SeqCst), 1);
    assert_eq!(Arc::strong_count(&counters), 1);

    // Reverse order: the record wrapper may be dropped first, while the inner
    // controller (and therefore the callback) is still owned by the caller.
    let counters = Arc::new(Counters::default());
    let controller =
        Controller::new_custom(DropProbe(counters.clone())).expect("new_custom should succeed");
    let record = Controller::new_record(&controller, recording.to_str().unwrap())
        .expect("new_record should succeed");
    drop(record);
    assert_eq!(
        counters.drops.load(Ordering::SeqCst),
        0,
        "dropping the record wrapper must not release a callback still owned by the inner controller"
    );
    assert!(controller.connected());
    drop(controller);
    assert_eq!(counters.drops.load(Ordering::SeqCst), 1);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn tasker_retains_custom_callback() {
    let counters = Arc::new(Counters::default());
    let controller =
        Controller::new_custom(DropProbe(counters.clone())).expect("new_custom should succeed");
    let tasker = Tasker::new().expect("tasker should be created");
    tasker
        .bind_controller(&controller)
        .expect("bind_controller should succeed");
    drop(controller);

    assert_eq!(
        counters.drops.load(Ordering::SeqCst),
        0,
        "tasker released the bound controller while still holding it"
    );
    assert!(
        tasker.controller().unwrap().connected(),
        "bound controller must stay usable through the tasker"
    );
    assert!(
        counters.connected_calls.load(Ordering::SeqCst) >= 1,
        "tasker did not reach the bound controller's connected callback"
    );

    drop(tasker);
    assert_eq!(counters.drops.load(Ordering::SeqCst), 1);
    assert_eq!(Arc::strong_count(&counters), 1);

    // Reverse order: destroying the tasker first must leave the caller's controller usable.
    let counters = Arc::new(Counters::default());
    let controller =
        Controller::new_custom(DropProbe(counters.clone())).expect("new_custom should succeed");
    let tasker = Tasker::new().expect("tasker should be created");
    tasker
        .bind_controller(&controller)
        .expect("bind_controller should succeed");
    drop(tasker);
    assert_eq!(
        counters.drops.load(Ordering::SeqCst),
        0,
        "destroying the tasker must not release a controller still owned by the caller"
    );
    assert!(controller.connected());
    drop(controller);
    assert_eq!(counters.drops.load(Ordering::SeqCst), 1);
}

#[test]
fn concurrent_last_owner_drops_release_callback_once() {
    const THREADS: usize = 8;

    for _ in 0..50 {
        let counters = Arc::new(Counters::default());
        let controller =
            Controller::new_custom(DropProbe(counters.clone())).expect("new_custom should succeed");
        let barrier = Arc::new(Barrier::new(THREADS + 1));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let clone = controller.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    drop(clone);
                })
            })
            .collect();

        barrier.wait();
        drop(controller);
        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counters.drops.load(Ordering::SeqCst), 1);
        assert_eq!(Arc::strong_count(&counters), 1);
    }
}

#[test]
fn concurrent_bind_controller_keeps_retention_in_sync() {
    let counters_a = Arc::new(Counters::default());
    let counters_b = Arc::new(Counters::default());
    let a =
        Controller::new_custom(DropProbe(counters_a.clone())).expect("new_custom should succeed");
    let b =
        Controller::new_custom(DropProbe(counters_b.clone())).expect("new_custom should succeed");
    let tasker = Tasker::new().expect("tasker should be created");

    for _ in 0..50 {
        let barrier = Arc::new(Barrier::new(3));
        let mut handles = Vec::new();
        for ctrl in [a.clone(), b.clone()] {
            let tasker = tasker.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                for _ in 0..20 {
                    tasker.bind_controller(&ctrl).unwrap();
                }
            }));
        }
        barrier.wait();
        for handle in handles {
            handle.join().unwrap();
        }
    }

    // bind_controller asserts that the native pointer and the retained clone agree;
    // reaching the bound controller also proves its callback is still alive.
    assert!(tasker.controller().unwrap().connected());
    assert_eq!(
        counters_a.drops.load(Ordering::SeqCst) + counters_b.drops.load(Ordering::SeqCst),
        0
    );

    drop(tasker);
    assert_eq!(
        counters_a.drops.load(Ordering::SeqCst) + counters_b.drops.load(Ordering::SeqCst),
        0,
        "caller handles still own both callbacks"
    );
    drop(a);
    assert_eq!(
        counters_a.drops.load(Ordering::SeqCst) + counters_b.drops.load(Ordering::SeqCst),
        1
    );
    drop(b);
    assert_eq!(
        counters_a.drops.load(Ordering::SeqCst) + counters_b.drops.load(Ordering::SeqCst),
        2
    );
}

#[test]
fn failed_connect_releases_callback_once() {
    let counters = Arc::new(Counters::default());
    let controller =
        Controller::new_custom(FailingProbe(counters.clone())).expect("new_custom should succeed");
    let id = controller
        .post_connection()
        .expect("post_connection should succeed");
    let status = controller.wait(id);
    assert!(
        status.is_failed(),
        "connect failure should surface as failed status"
    );
    assert_eq!(counters.connects.load(Ordering::SeqCst), 1);

    drop(controller);
    assert_eq!(counters.drops.load(Ordering::SeqCst), 1);
    assert_eq!(Arc::strong_count(&counters), 1);
}
