//! Small inference-focused helper crate.
//!
//! This crate is intentionally minimal today. Its role is to host narrow
//! runtime-facing traits without forcing downstream users to depend on the whole
//! training surface.

pub mod regressor;

pub use regressor::Regressor;
