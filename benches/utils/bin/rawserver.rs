// SPDX-License-Identifier: Apache-2.0
//! Raw-TCP, thread-per-connection search server — the leanest possible wire in
//! front of the embedded engine, for the throughput bench. No tonic/HTTP2/
//! protobuf/tokio: each client connection gets one OS thread that loops
//! read-query → vector_search → write-ids. This mirrors the native probe's
//! threading model (N threads calling vector_search), so it isolates the engine
//! throughput from gRPC serving overhead.
//!
//! Wire (little-endian, one persistent connection, pipelined requests):
//!   request : u32 k, u32 dim, then dim*4 bytes of f32 query
//!   response: u32 n, then n*16 bytes (each id = 16-byte big-endian decimal128)
//!
//! Bench tooling only; lives in the workspace-only bench crate.

use std::{
    env,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::Arc,
    thread,
};

use arrow_array::{Array, Decimal128Array};
use infino::{ConnectOptions, Supertable, connect_with};

const ID_BYTES: usize = 16;

fn envv(k: &str, d: &str) -> String {
    env::var(k).unwrap_or_else(|_| d.to_string())
}

fn handle(mut stream: TcpStream, st: Arc<Supertable>, col: String) {
    stream.set_nodelay(true).ok();
    let mut hdr = [0u8; 8];
    loop {
        if stream.read_exact(&mut hdr).is_err() {
            break; // client closed
        }
        let k = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
        let mut qbuf = vec![0u8; dim * 4];
        if stream.read_exact(&mut qbuf).is_err() {
            break;
        }
        let query: Vec<f32> = qbuf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let batches = match st.vector_search(&col, &query, k, None, None) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[rawserver] search error: {e}");
                break;
            }
        };
        let mut ids: Vec<u8> = Vec::with_capacity(k * ID_BYTES);
        for b in &batches {
            let Some(c) = b.column_by_name("_id") else { break };
            let Some(dec) = c.as_any().downcast_ref::<Decimal128Array>() else {
                break;
            };
            for i in 0..dec.len() {
                ids.extend_from_slice(&dec.value(i).to_be_bytes());
            }
        }
        let n = (ids.len() / ID_BYTES) as u32;
        let mut out = Vec::with_capacity(4 + ids.len());
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&ids);
        if stream.write_all(&out).is_err() {
            break;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = envv("PROBE_DATA", "/tmp/vectordb_bench/infino");
    let table = envv("PROBE_TABLE", "vdbbench_infino");
    let col = envv("PROBE_COL", "emb");
    let cache: u64 = env::var("PROBE_CACHE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(21_474_836_480);
    let addr = envv("RAW_ADDR", "0.0.0.0:50052");

    let opts = ConnectOptions::new()
        .with_cache_budget_bytes(cache)
        .with_cache_dir(format!("{data}/cache"));
    let conn = connect_with(&data, opts)?;
    let st = Arc::new(conn.open_table(&table)?);
    let warm = vec![0.0f32; 768];
    let _ = st.vector_search(&col, &warm, 10, None, None);

    let listener = TcpListener::bind(&addr)?;
    eprintln!("[rawserver] table={table} listening={addr} cache={cache} — ready");
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let st = st.clone();
                let col = col.clone();
                thread::spawn(move || handle(s, st, col));
            }
            Err(e) => eprintln!("[rawserver] accept error: {e}"),
        }
    }
    Ok(())
}
