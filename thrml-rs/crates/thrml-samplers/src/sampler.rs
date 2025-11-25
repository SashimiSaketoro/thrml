use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::node::TensorSpec;

pub trait AbstractConditionalSampler {
    /// Sample from the conditional distribution.
    ///
    /// - `interactions`: Sliced interaction tensors, each with shape [n_nodes, n_interactions, tail_dim]
    /// - `active_flags`: Boolean flags indicating which interactions are active, shape [n_nodes, n_interactions]
    /// - `neighbor_states`: Neighbor state tensors for each interaction, shape [n_nodes, n_interactions, ...]
    fn sample(
        &self,
        key: crate::rng::RngKey, // RNG key for deterministic sampling
        interactions: &[Tensor<WgpuBackend, 3>],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1>;

    fn init_state(&self) {}
}
