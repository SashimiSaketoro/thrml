use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::node::TensorSpec;
use crate::sampler::AbstractConditionalSampler;

pub struct BernoulliConditional;

impl AbstractConditionalSampler for BernoulliConditional {
    fn sample(
        &self,
        _key: crate::rng::RngKey,
        interactions: &[Tensor<WgpuBackend, 3>],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        // Compute gamma parameter (sum of interaction contributions)
        // Formula: gamma = sum_i (weights_i * active_i * spin_product_i)
        // where i iterates over interactions
        // The interaction tensor has shape [n_nodes, n_interactions, tail_dim]
        // We sum over the interaction dimension (axis 1) to get contributions per node
        let n_nodes = output_spec.shape[0];
        let mut gamma = Tensor::zeros([n_nodes], device);
        
        // Iterate through interactions and compute weighted sum
        for (interaction, active, _states) in itertools::izip!(interactions, active_flags, neighbor_states) {
            // The interaction tensor is [n_nodes, n_interactions, tail_dim]
            // For a simple case (bias-only), we can sum over the interaction and tail dimensions
            // For the full implementation, we need batch_gather and spin_product from DiscreteEBMFactor
            // Sum over dimension 1 (n_interactions), then squeeze -> [n_nodes, tail_dim]
            let interaction_sum_2d: Tensor<WgpuBackend, 2> = interaction.clone().sum_dim(1).squeeze_dim(1);
            // Sum over dimension 1 (tail_dim), then squeeze -> [n_nodes]
            let interaction_sum: Tensor<WgpuBackend, 1> = interaction_sum_2d.sum_dim(1).squeeze_dim(1);
            
            // Multiply by active flags (sum over interaction dimension)
            // Active is [n_nodes, n_interactions], sum over dimension 1, then squeeze -> [n_nodes]
            let active_sum: Tensor<WgpuBackend, 1> = active.clone().sum_dim(1).squeeze_dim(1);
            
            // Combine: gamma += interaction_sum * active_sum
            // This is a simplified version - the full version would use batch_gather and spin_product
            gamma = gamma + interaction_sum * active_sum;
        }
        
        // Sample: P(S=1) = sigmoid(2*gamma)
        let probs = burn::tensor::activation::sigmoid(gamma * 2.0);
        
        // Generate uniform random values in [0, 1] and threshold by probabilities
        // This implements Bernoulli sampling: sample ~ Uniform(0,1), then return (sample < prob)
        let uniform_random: Tensor<WgpuBackend, 1> = Tensor::random(
            probs.dims(),
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            device,
        );
        
        // Compare: sample < prob gives Bernoulli(prob)
        let samples_bool = uniform_random.lower_equal(probs);
        
        // Convert bool to the output dtype (should be Bool for spin nodes)
        // For now, return as float (will be cast as needed)
        samples_bool.float()
    }
}
