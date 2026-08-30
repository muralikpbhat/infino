// SPDX-License-Identifier: Apache-2.0
//! Loopback throughput bench server: wraps the embedded engine behind gRPC so a
//! multi-process client (e.g. the standard VDBBench MP runner) can drive it the
//! same way it drives any networked vector DB. This is BENCH TOOLING, not a
//! product daemon — it lives in the workspace-only bench crate and exists to
//! measure the engine's concurrency ceiling without the in-process Python GIL.
//!
//! One shared `Supertable` (one warm cache, GIL-free search). In-flight work is
//! bounded by a semaphore (~physical cores) so excess requests queue instead of
//! oversubscribing — that's what yields a clean rise-then-plateau under load.
//!
//! Config via env (mirrors examples/concurrent_probe.rs):
//!   PROBE_DATA   data path        (default /tmp/vectordb_bench/infino)
//!   PROBE_TABLE  table name       (default vdbbench_infino)
//!   PROBE_COL    vector column    (default emb)
//!   PROBE_CACHE  cache budget     (default 20 GiB)
//!   GRPC_ADDR    listen addr      (default 127.0.0.1:50051)
//!   INFINO_SERVER_INFLIGHT  max concurrent searches (default: num_cpus)
//! ef / search_mode come from $XDG_CONFIG_HOME/infino/config.yaml, as usual.

use std::{
    env,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use arrow_array::{Array, Decimal128Array};
use infino::{ConnectOptions, Supertable, connect_with};
use tokio::sync::Semaphore;
use tonic::{Request, Response, Status, transport::Server};

// Per-phase timing accumulators (microseconds), reset each reporting interval
// by the background logger. Purely diagnostic — locate where per-query time goes.
static CNT: AtomicU64 = AtomicU64::new(0);
static US_SEM: AtomicU64 = AtomicU64::new(0);
static US_DISPATCH: AtomicU64 = AtomicU64::new(0);
static US_DECODE: AtomicU64 = AtomicU64::new(0);
static US_SEARCH: AtomicU64 = AtomicU64::new(0);
static US_EXTRACT: AtomicU64 = AtomicU64::new(0);
static US_TOTAL: AtomicU64 = AtomicU64::new(0);

pub mod pb {
    tonic::include_proto!("infsearch");
}
use pb::search_server::{Search, SearchServer};
use pb::{SearchReply, SearchRequest};

const ID_BYTES: usize = 16;

struct SearchSvc {
    st: Arc<Supertable>,
    col: String,
    sem: Arc<Semaphore>,
}

#[tonic::async_trait]
impl Search for SearchSvc {
    async fn vector_search(
        &self,
        req: Request<SearchRequest>,
    ) -> Result<Response<SearchReply>, Status> {
        let t_start = Instant::now();
        let r = req.into_inner();
        let st = self.st.clone();
        let col = self.col.clone();
        let sem = self.sem.clone();
        let t_sem = Instant::now();
        let permit = sem
            .acquire_owned()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        let sem_us = t_sem.elapsed().as_micros() as u64;
        // vector_search is sync + CPU-bound; run it off the async workers.
        let t_spawn = Instant::now();
        let (ids, dispatch_us, decode_us, search_us, extract_us) =
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                let dispatch_us = t_spawn.elapsed().as_micros() as u64;
                // query arrives as raw little-endian f32 bytes (dim*4).
                let td = Instant::now();
                let query: Vec<f32> = r
                    .query
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let decode_us = td.elapsed().as_micros() as u64;
                let ts = Instant::now();
                let batches = st
                    .vector_search(&col, &query, r.k as usize, None, None)
                    .map_err(|e| e.to_string())?;
                let search_us = ts.elapsed().as_micros() as u64;
                let te = Instant::now();
                let mut bytes: Vec<u8> = Vec::with_capacity(r.k as usize * ID_BYTES);
                for b in &batches {
                    let c = b
                        .column_by_name("_id")
                        .ok_or_else(|| "result missing _id".to_string())?;
                    let dec = c
                        .as_any()
                        .downcast_ref::<Decimal128Array>()
                        .ok_or_else(|| "_id is not decimal128".to_string())?;
                    for i in 0..dec.len() {
                        bytes.extend_from_slice(&dec.value(i).to_be_bytes());
                    }
                }
                let extract_us = te.elapsed().as_micros() as u64;
                Ok::<_, String>((bytes, dispatch_us, decode_us, search_us, extract_us))
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::internal)?;

        let total_us = t_start.elapsed().as_micros() as u64;
        CNT.fetch_add(1, Ordering::Relaxed);
        US_SEM.fetch_add(sem_us, Ordering::Relaxed);
        US_DISPATCH.fetch_add(dispatch_us, Ordering::Relaxed);
        US_DECODE.fetch_add(decode_us, Ordering::Relaxed);
        US_SEARCH.fetch_add(search_us, Ordering::Relaxed);
        US_EXTRACT.fetch_add(extract_us, Ordering::Relaxed);
        US_TOTAL.fetch_add(total_us, Ordering::Relaxed);

        Ok(Response::new(SearchReply { ids }))
    }
}

fn envv(k: &str, d: &str) -> String {
    env::var(k).unwrap_or_else(|_| d.to_string())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cap the async I/O worker threads so tonic's runtime does not compete with
    // the engine's rayon reader pool for physical cores (the serving-tax fix).
    let io_threads: usize = env::var("INFINO_SERVER_IO_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(io_threads.max(1))
        .enable_all()
        .build()?;
    rt.block_on(run())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let data = envv("PROBE_DATA", "/tmp/vectordb_bench/infino");
    let table = envv("PROBE_TABLE", "vdbbench_infino");
    let col = envv("PROBE_COL", "emb");
    let cache: u64 = env::var("PROBE_CACHE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(21_474_836_480);
    let addr = envv("GRPC_ADDR", "127.0.0.1:50051");
    let inflight: usize = env::var("INFINO_SERVER_INFLIGHT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(num_cpus::get);

    let opts = ConnectOptions::new()
        .with_cache_budget_bytes(cache)
        .with_cache_dir(format!("{data}/cache"));
    let conn = connect_with(&data, opts)?;
    let st = Arc::new(conn.open_table(&table)?);

    // Warm the shared cache single-threaded before accepting load.
    let warm_q = vec![0.0f32; 768];
    let _ = st.vector_search(&col, &warm_q, 10, None, None);
    let io_threads = env::var("INFINO_SERVER_IO_THREADS").unwrap_or_else(|_| "2".into());
    eprintln!(
        "[grpcserver] table={table} listening={addr} inflight={inflight} io_threads={io_threads} cache={cache} — ready"
    );

    // Background per-phase reporter: every 5s, print average microseconds per
    // query in each stage (and reset), so we can see where the serving time goes.
    tokio::spawn(async {
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            let n = CNT.swap(0, Ordering::Relaxed);
            if n == 0 {
                continue;
            }
            let avg = |a: &AtomicU64| a.swap(0, Ordering::Relaxed) / n;
            eprintln!(
                "[phase] n={n} qps~{} | sem={}us dispatch={}us decode={}us search={}us extract={}us handler_total={}us",
                n / 5,
                avg(&US_SEM),
                avg(&US_DISPATCH),
                avg(&US_DECODE),
                avg(&US_SEARCH),
                avg(&US_EXTRACT),
                avg(&US_TOTAL),
            );
        }
    });

    let svc = SearchSvc {
        st,
        col,
        sem: Arc::new(Semaphore::new(inflight)),
    };
    Server::builder()
        .add_service(SearchServer::new(svc))
        .serve(addr.parse()?)
        .await?;
    Ok(())
}
