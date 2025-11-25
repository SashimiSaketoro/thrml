/// SpinGibbsConditional - Gibbs updates for spin-valued variables in discrete EBMs.
/// 
/// This sampler performs Gibbs sampling updates for spin (binary) variables,
/// computing the conditional distribution given neighboring states.
/// 
/// The conditional probability is: P(S=1) = sigmoid(2*γ)
/// where γ = Σ_i s_1^i ... s_K^i * W^i[x_1^i, ..., x_M^i]

use burn::tensor::{Tensor, Distribution};
use thrml_core::backend::WgpuBackend;
use thrml_core::node::TensorSpec;
use crate::sampler::AbstractConditionalSampler;
use crate::rng::RngKey;

/// A conditional update for spin-valued random variables that performs a Gibbs sampling update
/// given one or more DiscreteEBMInteractions.
pub struct SpinGibbsConditional;

impl SpinGibbsConditional {
    pub fn new() -> Self {
        SpinGibbsConditional
    }
}

impl Default for SpinGibbsConditional {
    fn default() -> Self {
        Self::new()
    }
}

impl AbstractConditionalSampler for SpinGibbsConditional {
    fn sample(
        &self,
        _key: RngKey,
        interactions: &[Tensor<WgpuBackend, 3>],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        // Compute gamma parameter
        let gamma = self.compute_parameters(
            interactions,
            active_flags,
            neighbor_states,
            output_spec,
            device,
        );
        
        // Sample: P(S=1) = sigmoid(2*gamma)
        let probs = burn::tensor::activation::sigmoid(gamma * 2.0);
        
        // Generate uniform random values and compare
        let n_nodes = output_spec.shape[0];
        let uniform: Tensor<WgpuBackend, 1> = Tensor::random(
            [n_nodes],
            Distribution::Uniform(0.0, 1.0),
            device,
        );
        
        // Sample Bernoulli: output 1 if uniform < probs, else 0
        uniform.lower_equal(probs).float()
    }
}

impl SpinGibbsConditional {
    /// Compute the parameter γ of a spin-valued Bernoulli distribution given DiscreteEBMInteractions.
    /// 
    /// γ = Σ_i s_1^i ... s_K^i * W^i[x_1^i, ..., x_M^i]
    /// 
    /// where the sum is over all the DiscreteEBMInteractions.
    fn compute_parameters(
        &self,
        interactions: &[Tensor<WgpuBackend, 3>],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        let n_nodes = output_spec.shape[0];
        let mut gamma: Tensor<WgpuBackend, 1> = Tensor::zeros([n_nodes], device);
        
        for (interaction, active, states) in 
            itertools::izip!(interactions, active_flags, neighbor_states) 
        {
            // interaction: [n_nodes, n_interactions, weight_dim]
            // active: [n_nodes, n_interactions]
            // states: Vec of [n_nodes, n_interactions] tensors (one per tail block)
            
            // For SpinGibbsConditional, we use the interaction.n_spin to determine
            // how many of the states are spin vs categorical.
            // The weights in the interaction tensor have already been gathered
            // based on categorical indices by the BlockSamplingProgram.
            
            // Get interaction dimensions
            let interaction_dims = interaction.dims();
            let _n_interactions = interaction_dims[1];
            
            // For spin variables, weights is 1D after categorical indexing
            // We need to compute: gamma += sum(weights * active * spin_prod, axis=-1)
            
            // Compute spin product from neighbor states
            // States are [n_nodes, n_interactions] - we compute element-wise product
            let spin_prod = compute_spin_product_2d(states, device);
            
            // The interaction tensor contains the weights (already indexed by categorical)
            // We need to sum over the interaction dimension
            // weights: [n_nodes, n_interactions, ...] -> need to reduce to [n_nodes, n_interactions]
            
            // For simplicity, if weights have extra dimensions, flatten them
            let weights = if interaction_dims.len() > 2 {
                // Sum over trailing dimensions to get [n_nodes, n_interactions]
                interaction.clone().sum_dim(2).squeeze_dim(2)
            } else {
                // Already [n_nodes, n_interactions]
                // Need to handle 3D tensor case
                interaction.clone().reshape([interaction_dims[0] as i32, interaction_dims[1] as i32])
            };
            
            // Compute contribution: weights * active * spin_prod, then sum over interaction dim
            let contribution = weights * active.clone() * spin_prod;
            let contribution_sum = contribution.sum_dim(1).squeeze_dim(1);
            
            gamma = gamma + contribution_sum;
        }
        
        gamma
    }
}

/// Compute the element-wise product of spin states.
/// 
/// For spin variables: True/1.0 -> +1, False/0.0 -> -1
/// Returns the element-wise product across all state tensors.
fn compute_spin_product_2d(
    states: &[Tensor<WgpuBackend, 2>],
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 2> {
    if states.is_empty() {
        // Return 1.0 tensor
        return Tensor::ones([1, 1], device);
    }
    
    // Convert each state to ±1: (2 * s - 1)
    let converted: Vec<Tensor<WgpuBackend, 2>> = states.iter()
        .map(|s| s.clone() * 2.0 - 1.0)
        .collect();
    
    // Compute element-wise product
    let first = converted[0].clone();
    converted.iter().skip(1).fold(first, |acc, x| acc * x.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[cfg(feature = "gpu")]
    #[test]
    fn test_spin_gibbs_basic() {
        use thrml_core::backend::{init_gpu_device, ensure_metal_backend};
        
        ensure_metal_backend();
        let device = init_gpu_device();
        
        let sampler = SpinGibbsConditional::new();
        
        // Create simple test case
        let output_spec = TensorSpec {
            shape: vec![4],
            dtype: burn::tensor::DType::Bool,
        };
        
        // Empty interactions -> gamma = 0 -> P(S=1) = 0.5
        let interactions: Vec<Tensor<WgpuBackend, 3>> = Vec::new();
        let active_flags: Vec<Tensor<WgpuBackend, 2>> = Vec::new();
        let neighbor_states: Vec<Vec<Tensor<WgpuBackend, 2>>> = Vec::new();
        
        let key = RngKey::new(42);
        let samples = sampler.sample(key, &interactions, &active_flags, &neighbor_states, &output_spec, &device);
        
        assert_eq!(samples.dims(), [4], "Should produce 4 samples");
    }
}

