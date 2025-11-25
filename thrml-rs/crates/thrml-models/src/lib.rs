//! # thrml-models
//!
//! Model implementations for the THRML probabilistic computing library.
//!
//! This crate provides implementations of various probabilistic graphical models:
//!
//! ## Ising Model
//!
//! The classic Ising model for spin systems:
//!
//! ```rust,ignore
//! use thrml_models::ising::{IsingEBM, IsingSamplingProgram, hinton_init};
//!
//! let model = IsingEBM::new(nodes, edges, biases, weights, beta);
//! let energy = model.energy(&state, &blocks, &device);
//! ```
//!
//! ## Discrete EBM Factors
//!
//! Flexible factor types for building custom energy-based models:
//!
//! - [`SpinEBMFactor`]: Binary/spin interactions
//! - [`CategoricalEBMFactor`]: Categorical variable interactions
//! - [`DiscreteEBMFactor`]: Mixed spin + categorical
//! - [`SquareDiscreteEBMFactor`]: Symmetric interactions
//!
//! ## Training Utilities
//!
//! - [`ising::estimate_moments`]: Estimate first/second moments via sampling
//! - [`ising::estimate_kl_grad`]: Estimate KL divergence gradients
//! - [`ising::hinton_init`]: Initialize states from marginal biases

#![recursion_limit = "256"]  // Required for burn-wgpu

pub mod factor;
pub mod ebm;
pub mod discrete_ebm;
pub mod ising;

pub use factor::*;
pub use ebm::*;
pub use discrete_ebm::*;
pub use ising::*;

