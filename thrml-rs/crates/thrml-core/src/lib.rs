//! # thrml-core
//!
//! Core types and GPU backend for the THRML probabilistic computing library.
//!
//! This crate provides foundational types for building probabilistic graphical models:
//!
//! - [`Node`]: Represents a random variable in the graph
//! - [`NodeType`]: Spin (binary ±1) or Categorical variables  
//! - [`Block`]: A collection of nodes of the same type
//! - [`BlockSpec`]: Specification for mapping between local and global state
//! - [`InteractionGroup`]: Defines interactions between node groups
//!
//! ## GPU Backend
//!
//! The `gpu` feature provides GPU acceleration via WGPU:
//!
//! ```rust,ignore
//! use thrml_core::backend::{init_gpu_device, ensure_metal_backend};
//!
//! ensure_metal_backend();
//! let device = init_gpu_device();
//! ```

#![recursion_limit = "256"]  // Required for burn-wgpu

pub mod backend;
pub mod node;
pub mod block;
pub mod blockspec;
pub mod state_tree;
pub mod interaction;

pub use backend::*;
pub use node::*;
pub use block::*;
pub use blockspec::*;
pub use state_tree::*;
pub use interaction::*;

