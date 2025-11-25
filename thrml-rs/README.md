# thrml-rs

**GPU-accelerated probabilistic graphical models in Rust**

[![Crates.io](https://img.shields.io/crates/v/thrml.svg)](https://crates.io/crates/thrml)
[![Documentation](https://docs.rs/thrml/badge.svg)](https://docs.rs/thrml)
[![License](https://img.shields.io/crates/l/thrml.svg)](LICENSE-MIT)

`thrml-rs` is a Rust implementation of the [THRML](https://github.com/extropic-ai/thrml) probabilistic computing library, 
providing GPU-accelerated sampling for probabilistic graphical models (PGMs).

## Features

- 🚀 **GPU Acceleration**: Native Metal support on macOS via WGPU
- 🎲 **Block Gibbs Sampling**: Efficient parallel sampling for PGMs
- 🧠 **Energy-Based Models**: Ising models, discrete EBMs, and more
- 🔢 **Categorical & Spin Variables**: Full support for mixed discrete models
- 🔄 **Deterministic RNG**: Reproducible sampling with ChaCha8-based key splitting
- 📊 **Moment Estimation**: Built-in observers for computing statistics

## Quick Start

```rust
use thrml_core::{Node, NodeType, Block};
use thrml_models::ising::{IsingEBM, IsingSamplingProgram, hinton_init};
use thrml_samplers::{RngKey, SamplingSchedule};
use burn::tensor::Tensor;

// Create a 5-node Ising chain
let nodes: Vec<Node> = (0..5).map(|_| Node::new(NodeType::Spin)).collect();
let edges: Vec<_> = nodes.windows(2)
    .map(|w| (w[0].clone(), w[1].clone()))
    .collect();

// Define biases and coupling weights
let device = thrml_core::backend::init_gpu_device();
let biases = Tensor::from_data(vec![0.1f32, 0.2, 0.0, -0.1, 0.3].as_slice(), &device);
let weights = Tensor::from_data(vec![0.5f32, -0.3, 0.4, 0.2].as_slice(), &device);
let beta = Tensor::from_data(vec![1.0f32].as_slice(), &device);

// Create the Ising model
let model = IsingEBM::new(nodes.clone(), edges, biases, weights, beta);

// Initialize using Hinton's method
let key = RngKey::new(42);
let blocks = vec![Block::new(nodes).unwrap()];
let init_state = hinton_init(key, &model, &blocks, &[], &device);
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| `thrml-core` | Core types: Node, Block, BlockSpec, GPU backend |
| `thrml-samplers` | Sampling algorithms: Gibbs, Bernoulli, Softmax |
| `thrml-models` | Model implementations: Ising, Discrete EBM |
| `thrml-observers` | Observation utilities: State, Moments |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
thrml = "0.1.3"

# Or individual crates:
thrml-core = "0.1.3"
thrml-samplers = "0.1.3"
thrml-models = "0.1.3"
```

### Feature Flags

- `gpu` (default): Enable GPU acceleration via WGPU/Metal

## Requirements

- Rust 1.75+
- GPU with Metal support (macOS) or Vulkan (Linux/Windows)

## Examples

See the [`examples/`](crates/thrml-examples/examples/) directory:

```bash
cargo run --example ising_chain --features gpu
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

