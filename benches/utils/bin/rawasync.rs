// SPDX-License-Identifier: Apache-2.0
//! Async raw-TCP search server: tokio epoll reactor + raw length-framed
//! messages (no HTTP2, no protobuf) + spawn_blocking search. Unlike the
//! thread-per-connection rawserver, all connections are driven by a small pool
//! of tokio I/O workers (one lightweight task per connection), so 64 clients
//! cost ~cores threads, not 64 OS threads waking/sleeping on blocking sockets.
//! This targets the network/socket CPU that dominates the 2-box serving tax.
//!
//! Wire (little-endian, persistent connection, pipelined):
//!   request : u32 k, u32 dim, then dim*4 bytes of f32 query
//!   response: u32 n, then n*16 bytes (each id = 16-byte big-endian decimal128)
//!
//! Env: PROBE_DATA, PROBE_TABLE, PROBE_COL, PROBE_CACHE, RAW_ADDR,
//!      INFINO_SERVER_IO_THREADS (tokio workers, default 4),
//!      INFINO_SERVER_INFLIGHT   (max concurrent searches, default num_cpus).
//! Bench tooling only (workspace-only crate).

use std::{env, sync::Arc};

use arrow_array::{Array, Decimal128Array};
use infino::{ConnectOptions, Supertable, connect_with};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::Semaphore,
};

const ID_BYTES: usize = 16;

fn envv(k: &str, d: &str) -> String {
    env::var(k).unwrap_or_else(|_| d.to_string())
}

async fn handle(mut stream: TcpStream, st: Arc<Supertable>, col: String, sem: Arc<Semaphore>) {
    stream.set_nodelay(true).ok();
    let mut hdr = [0u8; 8];
    loop {
        if stream.read_exact(&mut hdr).await.is_err() {
            break; // client closed
        }
        let k = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
        let mut qbuf = vec![0u8; dim * 4];
        if stream.read_exact(&mut qbuf).await.is_err() {
            break;
        }
        let st2 = st.clone();
        let col2 = col.clone();
        let permit = match sem.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => break,
        };
        let out = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let query: Vec<f32> = qbuf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let batches = st2.vector_search(&col2, &query, k, None, None).ok()?;
            let mut ids: Vec<u8> = Vec::with_capacity(k * ID_BYTES);
            for b in &batches {
                let c = b.column_by_name("_id")?;
                let dec = c.as_any().downcast_ref::<Decimal128Array>()?;
                for i in 0..dec.len() {
                    ids.extend_from_slice(&dec.value(i).to_be_bytes());
                }
            }
            let n = (ids.len() / ID_BYTES) as u32;
            let mut out = Vec::with_capacity(4 + ids.len());
            out.extend_from_slice(&n.to_le_bytes());
            out.extend_from_slice(&ids);
            Some(out)
        })
        .await;
        let out = match out {
            Ok(Some(o)) => o,
            _ => break,
        };
        if stream.write_all(&out).await.is_err() {
            break;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let io_threads: usize = env::var("INFINO_SERVER_IO_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
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
    let addr = envv("RAW_ADDR", "0.0.0.0:50053");
    let inflight: usize = env::var("INFINO_SERVER_INFLIGHT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(num_cpus::get);

    let opts = ConnectOptions::new()
        .with_cache_budget_bytes(cache)
        .with_cache_dir(format!("{data}/cache"));
    let conn = connect_with(&data, opts)?;
    let st = Arc::new(conn.open_table(&table)?);
    let warm = vec![0.0f32; 768];
    let _ = st.vector_search(&col, &warm, 10, None, None);

    let sem = Arc::new(Semaphore::new(inflight));
    let listener = TcpListener::bind(&addr).await?;
    let io = env::var("INFINO_SERVER_IO_THREADS").unwrap_or_else(|_| "4".into());
    eprintln!(
        "[rawasync] table={table} listening={addr} io_threads={io} inflight={inflight} cache={cache} — ready"
    );
    loop {
        let (stream, _) = listener.accept().await?;
        let st = st.clone();
        let col = col.clone();
        let sem = sem.clone();
        tokio::spawn(handle(stream, st, col, sem));
    }
}
