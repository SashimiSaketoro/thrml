# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2025-11-25

### Added

- Initial Rust port of the Python THRML library
- **thrml-core**: Core types (Node, Block, BlockSpec, InteractionGroup)
- **thrml-samplers**: Sampling algorithms
  - Block Gibbs sampling with `BlockSamplingProgram`
  - `BernoulliConditional` sampler for spin variables
  - `SoftmaxConditional` sampler using Gumbel-max trick
  - `SpinGibbsConditional` for Ising-type models
  - `CategoricalGibbsConditional` for categorical variables
  - Deterministic RNG key system (`RngKey`)
- **thrml-models**: Model implementations
  - `IsingEBM` with energy computation and factor generation
  - `IsingSamplingProgram` for Ising model sampling
  - `DiscreteEBMFactor` with multi-dimensional `batch_gather`
  - Specialized factors: `SpinEBMFactor`, `CategoricalEBMFactor`
  - Training utilities: `estimate_moments`, `estimate_kl_grad`, `hinton_init`
- **thrml-observers**: Observation utilities
  - `StateObserver` for collecting state samples
  - `MomentAccumulatorObserver` for moment statistics
- GPU acceleration via WGPU/Metal backend

### Technical Notes

- Uses Burn 0.19 for GPU tensor operations
- ChaCha8-based deterministic RNG for reproducibility
- Linear indexing for multi-dimensional tensor gather operations
- Full feature parity with Python THRML

## [Unreleased]

### Planned

- Performance benchmarks vs Python/JAX
- Additional examples (MNIST RBM)
- Python bindings via PyO3

