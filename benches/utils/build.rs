// SPDX-License-Identifier: Apache-2.0
//! Codegen for the loopback bench server's gRPC service (proto/search.proto).
//! This crate is workspace-only (publish = false), so tonic/prost never enter
//! the published `infino` crate's dependency tree. Uses a vendored protoc so no
//! system protobuf-compiler install is required.
fn main() {
    println!("cargo:rerun-if-changed=proto/search.proto");
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc binary");
    // SAFETY: single-threaded build script; no other thread reads the env here.
    unsafe {
        std::env::set_var("PROTOC", &protoc);
    }
    tonic_build::compile_protos("proto/search.proto").expect("compile proto/search.proto");
}
