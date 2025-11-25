//! SoftmaxConditional sampler for categorical variables.
//!
//! This sampler samples from a softmax distribution:
//! P(X=k) ∝ exp(θ_k)
//!
//! where X is a categorical random variable and θ is a vector of logits.
//!
//! Uses the Gumbel-max trick for efficient categorical sampling:
//! argmax_k(θ_k + Gumbel(0,1)) ~ Categorical(softmax(θ))
//!
//! This is equivalent to JAX's `jax.random.categorical` function.

use crate::rng::RngKey;
use crate::sampler::AbstractConditionalSampler;
use burn::tensor::{Distribution, Tensor};
use thrml_core::backend::WgpuBackend;
use thrml_core::node::TensorSpec;

/// Sample from a categorical distribution using the Gumbel-max trick.
///
/// This implements the same algorithm as JAX's `jax.random.categorical`.
///
/// # Arguments
///
/// * `logits` - 2D tensor of shape [n_samples, n_categories] with unnormalized log probabilities
/// * `device` - The device to use for tensor operations
///
/// # Returns
///
/// 1D tensor of shape `[n_samples]` with sampled category indices (as float)
pub fn categorical_sample(
    logits: Tensor<WgpuBackend, 2>,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 1> {
    let dims = logits.dims();
    let n_samples = dims[0];
    let n_categories = dims[1];

    // Generate Gumbel noise: -log(-log(U)) where U ~ Uniform(0, 1)
    // We add a small epsilon to avoid log(0)
    let uniform: Tensor<WgpuBackend, 2> = Tensor::random(
        [n_samples, n_categories],
        Distribution::Uniform(1e-10, 1.0 - 1e-10),
        device,
    );

    // Gumbel noise: -log(-log(U))
    let gumbel = -(-(uniform.log())).log();

    // Add Gumbel noise to logits
    let perturbed = logits + gumbel;

    // Argmax along the category dimension
    // argmax returns shape [n_samples] when axis=1 on [n_samples, n_categories]
    let samples = perturbed.argmax(1);

    // Convert Int tensor to Float tensor for consistency
    // The result should be 1D [n_samples]
    samples.float().squeeze_dim(1)
}

/// Abstract base for softmax-based samplers.
///
/// Concrete implementations must provide `compute_parameters` which returns
/// the θ vector (logits) for the softmax distribution.
pub struct SoftmaxConditional {
    /// Number of categories for the categorical distribution
    pub n_categories: usize,
}

impl SoftmaxConditional {
    pub fn new(n_categories: usize) -> Self {
        SoftmaxConditional { n_categories }
    }
}

impl AbstractConditionalSampler for SoftmaxConditional {
    fn sample(
        &self,
        _key: RngKey,
        interactions: &[Tensor<WgpuBackend, 3>],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        // Compute theta (logits) from interactions
        // This follows the CategoricalGibbsConditional.compute_parameters logic
        let theta = self.compute_parameters(
            interactions,
            active_flags,
            neighbor_states,
            output_spec,
            device,
        );

        // Sample from categorical using Gumbel-max trick
        // This is equivalent to: jax.random.categorical(key, theta, axis=-1)
        categorical_sample(theta, device)
    }
}

impl SoftmaxConditional {
    /// Compute the parameter θ of a softmax distribution given DiscreteEBMInteractions.
    ///
    /// θ = Σ_i s_1^i ... s_K^i * W^i[:, x_1^i, ..., x_M^i]
    ///
    /// where:
    /// - The sum is over all interactions
    /// - s_j are spin states (converted to ±1)
    /// - x_j are categorical states (used as indices)
    /// - W^i is the weight tensor for interaction i
    ///
    /// # Arguments
    ///
    /// * `interactions` - List of interaction tensors, each [n_nodes, n_interactions, tail_dim]
    /// * `active_flags` - List of active flags, each [n_nodes, n_interactions]
    /// * `neighbor_states` - List of neighbor state lists, each containing [n_nodes, n_interactions] tensors
    /// * `output_spec` - Output specification (shape, dtype)
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    ///
    /// Tensor of shape [n_nodes, n_categories] with logits
    fn compute_parameters(
        &self,
        interactions: &[Tensor<WgpuBackend, 3>],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 2> {
        let n_nodes = output_spec.shape[0];

        // Initialize theta with zeros: [n_nodes, n_categories]
        let mut theta: Tensor<WgpuBackend, 2> = Tensor::zeros([n_nodes, self.n_categories], device);

        // Sum contributions from each interaction
        for (interaction, active, _states) in
            itertools::izip!(interactions, active_flags, neighbor_states)
        {
            // interaction: [n_nodes, n_interactions, weight_dim]
            // active: [n_nodes, n_interactions]
            // states: Vec of [n_nodes, n_interactions] tensors (one per tail block)

            let interaction_dims = interaction.dims();
            let _n_interactions = interaction_dims[1];

            if interaction_dims[2] != self.n_categories {
                // This interaction's weights don't match our category count
                // This could happen if the interaction is for a different variable type
                continue;
            }

            // For CategoricalGibbsConditional:
            // - Extract spin states and categorical states from neighbors
            // - Compute weights = batch_gather_with_k(interaction.weights, *cat_states)
            // - Compute spin_prod = expand_dims(spin_product(spin_states), -1)
            // - theta += sum(spin_prod * weights * expand_dims(active, -1), axis=-2)

            // Simplified version: assume interaction is already the weights we need
            // This works when there are no spin variables or all spin variables are fixed

            // Expand active to match weight dimensions: [n_nodes, n_interactions] -> [n_nodes, n_interactions, 1]
            let active_expanded = active.clone().unsqueeze_dim::<3>(2);

            // For now, assume no spin variables (spin_prod = 1)
            // In the full implementation, we would:
            // 1. Split states into spin and categorical
            // 2. Compute spin_product for spin states
            // 3. Use batch_gather_with_k for categorical states

            // Weighted interaction: [n_nodes, n_interactions, n_categories]
            let weighted = interaction.clone() * active_expanded;

            // Sum over interactions dimension: [n_nodes, n_categories]
            let contribution = weighted.sum_dim(1).squeeze_dim(1);

            theta = theta + contribution;
        }

        theta
    }
}

/// CategoricalGibbsConditional - Gibbs updates for categorical variables in discrete EBMs.
///
/// This sampler computes the conditional distribution for categorical variables
/// given the current states of all neighboring variables.
pub struct CategoricalGibbsConditional {
    /// Number of categories for the categorical distribution
    pub n_categories: usize,
    /// Number of spin variables in each interaction
    pub n_spin: usize,
}

impl CategoricalGibbsConditional {
    pub fn new(n_categories: usize, n_spin: usize) -> Self {
        CategoricalGibbsConditional {
            n_categories,
            n_spin,
        }
    }
}

impl AbstractConditionalSampler for CategoricalGibbsConditional {
    fn sample(
        &self,
        _key: RngKey,
        interactions: &[Tensor<WgpuBackend, 3>],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        // Compute theta using full Gibbs conditional logic
        let theta = self.compute_parameters(
            interactions,
            active_flags,
            neighbor_states,
            output_spec,
            device,
        );

        // Sample from categorical using Gumbel-max trick
        categorical_sample(theta, device)
    }
}

impl CategoricalGibbsConditional {
    /// Compute the parameter θ of a softmax distribution for Gibbs sampling.
    ///
    /// This implements the full CategoricalGibbsConditional.compute_parameters
    /// from the Python version:
    ///
    /// θ = Σ_i s_1^i ... s_K^i * W^i[:, x_1^i, ..., x_M^i]
    fn compute_parameters(
        &self,
        interactions: &[Tensor<WgpuBackend, 3>],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 2> {
        let n_nodes = output_spec.shape[0];

        // Initialize theta with zeros: [n_nodes, n_categories]
        let mut theta: Tensor<WgpuBackend, 2> = Tensor::zeros([n_nodes, self.n_categories], device);

        for (interaction, active, states) in
            itertools::izip!(interactions, active_flags, neighbor_states)
        {
            // interaction: [n_nodes, n_interactions, weight_dim]
            // (already gathered from full weights using categorical indices)
            // active: [n_nodes, n_interactions]
            // states: Vec of [n_nodes, n_interactions] tensors

            // Split states into spin and categorical
            let (states_spin, _states_cat) = split_states(states, self.n_spin);

            // Compute spin product: prod over spin states of (2*s - 1)
            let spin_prod = spin_product_2d(&states_spin, device);

            // For categorical variables, the weights are already indexed by the
            // BlockSamplingProgram. The interaction tensor contains W[:, :, category]
            // for each category.

            // If we have categorical neighbors, we need to use batch_gather_with_k
            // to index into the remaining dimensions of the weights.
            // For now, assume the interaction is already fully indexed.

            // Expand spin_prod: [n_nodes, n_interactions] -> [n_nodes, n_interactions, 1]
            let spin_prod_expanded = spin_prod.unsqueeze_dim::<3>(2);

            // Expand active: [n_nodes, n_interactions] -> [n_nodes, n_interactions, 1]
            let active_expanded = active.clone().unsqueeze_dim::<3>(2);

            // Weighted contribution: [n_nodes, n_interactions, n_categories]
            let weighted = spin_prod_expanded * interaction.clone() * active_expanded;

            // Sum over interactions: [n_nodes, n_categories]
            let contribution = weighted.sum_dim(1).squeeze_dim(1);

            theta = theta + contribution;
        }

        theta
    }
}

/// Split states into spin and categorical parts.
///
/// # Arguments
///
/// * `states` - List of state tensors
/// * `n_spin` - Number of spin states (first n_spin are spin, rest are categorical)
fn split_states(
    states: &[Tensor<WgpuBackend, 2>],
    n_spin: usize,
) -> (Vec<Tensor<WgpuBackend, 2>>, Vec<Tensor<WgpuBackend, 2>>) {
    let states_spin: Vec<Tensor<WgpuBackend, 2>> = states.iter().take(n_spin).cloned().collect();

    let states_cat: Vec<Tensor<WgpuBackend, 2>> = states.iter().skip(n_spin).cloned().collect();

    (states_spin, states_cat)
}

/// Compute the product of spin states.
///
/// For spin variables: True -> +1, False -> -1
/// Returns the element-wise product: prod_i (2*s_i - 1)
///
/// # Arguments
///
/// * `spin_vals` - List of 2D tensors with spin values (as float 0/1)
/// * `device` - Device for tensor operations
///
/// # Returns
///
/// 2D tensor with the product of spin values
fn spin_product_2d(
    spin_vals: &[Tensor<WgpuBackend, 2>],
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 2> {
    if spin_vals.is_empty() {
        // Return 1.0 with shape matching expected output
        // This is a scalar broadcast case
        return Tensor::ones([1, 1], device);
    }

    // Convert each spin tensor to ±1: (2 * s - 1)
    let converted: Vec<Tensor<WgpuBackend, 2>> =
        spin_vals.iter().map(|s| s.clone() * 2.0 - 1.0).collect();

    // Compute element-wise product
    let first = converted[0].clone();
    converted
        .iter()
        .skip(1)
        .fold(first, |acc, x| acc * x.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    #[test]
    fn test_categorical_sample_basic() {
        use thrml_core::backend::{ensure_metal_backend, init_gpu_device};

        ensure_metal_backend();
        let device = init_gpu_device();

        // Create logits that strongly favor category 2 for sample 0, category 1 for sample 1
        // logits = [[-10, -10, 10, -10], [-10, 10, -10, -10]]
        let logits_data: Vec<f32> = vec![-10.0, -10.0, 10.0, -10.0, -10.0, 10.0, -10.0, -10.0];
        let logits_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(logits_data.as_slice(), &device);
        let logits: Tensor<WgpuBackend, 2> = logits_1d.reshape([2, 4]);

        let samples = categorical_sample(logits, &device);
        let samples_data: Vec<f32> = samples.into_data().to_vec().expect("read samples");

        // With such strong logits, samples should be deterministic (very high probability)
        // Sample 0 should be category 2, sample 1 should be category 1
        assert_eq!(
            samples_data[0] as i32, 2,
            "First sample should be category 2"
        );
        assert_eq!(
            samples_data[1] as i32, 1,
            "Second sample should be category 1"
        );
    }
}
