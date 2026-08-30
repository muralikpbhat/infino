// SPDX-License-Identifier: Apache-2.0
//! Native concurrent vector-search probe: opens an existing table and sweeps
//! request concurrency with plain OS threads (no Python, no marshalling), to
//! isolate the engine's raw concurrency ceiling from any harness overhead.
//!
//! Config (ef / search_mode) is read from $XDG_CONFIG_HOME/infino/config.yaml,
//! exactly as the embedded client does. Queries are a flat little-endian f32
//! file (n_queries * dim), e.g. dumped from the dataset's test split.

use std::{
    env, fs,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use infino::{ConnectOptions, connect_with};

fn envv(k: &str, d: &str) -> String {
    env::var(k).unwrap_or_else(|_| d.to_string())
}
fn envn(k: &str, d: usize) -> usize {
    env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn main() {
    let data_path = envv("PROBE_DATA", "/tmp/vectordb_bench/infino");
    let table = envv("PROBE_TABLE", "vdbbench_infino");
    let qfile = envv("PROBE_QUERIES", "/data/queries_1m.f32");
    let col = envv("PROBE_COL", "emb");
    let dim = envn("PROBE_DIM", 768);
    let k = envn("PROBE_K", 100);
    let secs = envn("PROBE_SECS", 8) as u64;
    let warmup = envn("PROBE_WARMUP", 2) as u64;
    let cache: u64 = envn("PROBE_CACHE", 21_474_836_480) as u64;

    // Load queries: flat LE f32, n * dim.
    let raw = fs::read(&qfile).expect("read query file");
    let nq = raw.len() / 4 / dim;
    assert!(nq > 0, "no queries parsed (file {} bytes, dim {})", raw.len(), dim);
    let mut queries: Vec<Vec<f32>> = Vec::with_capacity(nq);
    for qi in 0..nq {
        let mut v = Vec::with_capacity(dim);
        for d in 0..dim {
            let o = (qi * dim + d) * 4;
            v.push(f32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]));
        }
        queries.push(v);
    }
    let queries = Arc::new(queries);
    eprintln!("[probe] loaded {nq} queries dim={dim}; table={table} k={k}");

    let opts = ConnectOptions::new()
        .with_cache_budget_bytes(cache)
        .with_cache_dir(format!("{data_path}/cache"));
    let conn = connect_with(&data_path, opts).expect("connect");
    let st = Arc::new(conn.open_table(&table).expect("open_table"));

    // Warm the in-process caches single-threaded before timing.
    let warm_n = queries.len().min(2000);
    let wt = Instant::now();
    for q in queries.iter().take(warm_n) {
        let _ = st.vector_search(&col, q, k, None, None).expect("warm search");
    }
    eprintln!("[probe] warmed {warm_n} queries in {:.1}s", wt.elapsed().as_secs_f64());

    let levels: Vec<usize> = envv("PROBE_LEVELS", "1,6,8,9,10,11,12,14,16,20,32")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("conc,qps,p50_ms,p95_ms,p99_ms,count");
    for &n in &levels {
        let stop = Arc::new(AtomicBool::new(false));
        let phase_start = Instant::now();
        let mut handles = Vec::with_capacity(n);
        for t in 0..n {
            let st = st.clone();
            let q = queries.clone();
            let stop = stop.clone();
            let col = col.clone();
            handles.push(thread::spawn(move || {
                let mut lat: Vec<Duration> = Vec::new();
                let mut i = t;
                let ql = q.len();
                while !stop.load(Ordering::Relaxed) {
                    let query = &q[i % ql];
                    i += 1;
                    let t0 = Instant::now();
                    let _ = st.vector_search(&col, query, k, None, None).expect("search");
                    if phase_start.elapsed() > Duration::from_secs(warmup) {
                        lat.push(t0.elapsed());
                    }
                }
                lat
            }));
        }
        thread::sleep(Duration::from_secs(warmup + secs));
        stop.store(true, Ordering::Relaxed);
        let mut all: Vec<Duration> = Vec::new();
        for h in handles {
            all.extend(h.join().expect("join"));
        }
        all.sort_unstable();
        let count = all.len();
        let qps = count as f64 / secs as f64;
        let pct = |p: f64| {
            if count == 0 {
                0.0
            } else {
                let idx = ((count as f64 * p) as usize).min(count - 1);
                all[idx].as_secs_f64() * 1000.0
            }
        };
        println!(
            "{n},{:.0},{:.3},{:.3},{:.3},{count}",
            qps,
            pct(0.50),
            pct(0.95),
            pct(0.99)
        );
    }
    eprintln!("[probe] done");
}
