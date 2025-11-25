//! # thrml-samplers
//!
//! Sampling algorithms for the THRML probabilistic computing library.
//!
//! This crate provides GPU-accelerated sampling algorithms for probabilistic graphical models:
//!
//! - **Block Gibbs Sampling**: Parallel sampling of independent blocks via [`BlockSamplingProgram`]
//! - **Bernoulli Sampler**: For binary/spin variables via [`BernoulliConditional`]
//! - **Softmax Sampler**: For categorical variables using Gumbel-max trick via [`SoftmaxConditional`]
//! - **Spin Gibbs Sampler**: Specialized for Ising-type models via [`SpinGibbsConditional`]
//!
//! ## RNG Key System
//!
//! Deterministic RNG key management (similar to JAX):
//!
//! ```rust
//! use thrml_samplers::RngKey;
//!
//! let key = RngKey::new(42);
//! let (key1, key2) = key.split_two();
//! ```
//!
//! ## Sampling Schedule
//!
//! Control warmup, number of samples, and steps per sample:
//!
//! ```rust
//! use thrml_samplers::SamplingSchedule;
//!
//! let schedule = SamplingSchedule::new(100, 1000, 5);
//! // 100 warmup steps, 1000 samples, 5 steps between samples
//! ```

#![recursion_limit = "256"]  // Required for burn-wgpu

pub mod sampler;
pub mod bernoulli;
pub mod softmax;
pub mod spin_gibbs;
pub mod program;
pub mod schedule;
pub mod rng;
pub mod sampling;

pub use sampler::*;
pub use bernoulli::*;
pub use softmax::*;
pub use spin_gibbs::*;
pub use program::*;
pub use schedule::*;
pub use rng::*;
pub use sampling::*;
