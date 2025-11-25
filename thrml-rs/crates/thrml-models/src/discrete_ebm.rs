use crate::ebm::EBMFactor;
use crate::factor::{AbstractFactor, FactorInteractionGroup};
use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::block::Block;
use thrml_core::blockspec::BlockSpec;
use thrml_core::node::Node;
use thrml_core::state_tree::from_global_state;

/// Type alias for spin state tensors (boolean 2D tensors)
pub type SpinStates = Vec<Tensor<WgpuBackend, 2, burn::tensor::Bool>>;
/// Type alias for categorical state tensors (integer 2D tensors)
pub type CatStates = Vec<Tensor<WgpuBackend, 2, burn::tensor::Int>>;

/// An interaction that shows up when sampling from discrete-variable EBMs.
#[derive(Clone)]
pub struct DiscreteEBMInteraction {
    /// Number of spin states involved in the interaction
    pub n_spin: usize,
    /// Weight tensor associated with this interaction [batch, ..., dims]
    pub weights: Tensor<WgpuBackend, 3>,
}

impl DiscreteEBMInteraction {
    pub fn new(n_spin: usize, weights: Tensor<WgpuBackend, 3>) -> Self {
        DiscreteEBMInteraction { n_spin, weights }
    }
}

/// Multiply spin values (convert bool to -1/+1)
pub fn spin_product(
    spin_vals: &[Tensor<WgpuBackend, 1, burn::tensor::Bool>],
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 1> {
    if spin_vals.is_empty() {
        return Tensor::ones([1], device); // Return 1.0
    }
    // Convert bool to f32: True -> 1.0, False -> -1.0
    let converted: Vec<Tensor<WgpuBackend, 1>> = spin_vals
        .iter()
        .map(|v| {
            let float_tensor: Tensor<WgpuBackend, 1> = v.clone().float();
            float_tensor * 2.0 - 1.0
        })
        .collect();
    // Multiply all together - need to dereference for multiplication
    let first = converted[0].clone();
    converted
        .iter()
        .skip(1)
        .fold(first, |acc, x| acc * x.clone())
}

/// Index into weight tensor using categorical indices
///
/// This implements multi-dimensional advanced indexing using linear indexing.
/// The weights tensor has shape [batch, dim1, dim2, ..., dimN] where:
/// - The first dimension is the batch dimension
/// - The remaining dimensions correspond to categorical indices
///
/// The function computes flat indices using strides and then uses a single
/// `select` operation on the flattened tensor for efficiency.
///
/// # Arguments
///
/// * `weights` - 3D tensor with shape [batch, dim1, dim2, ...]
/// * `indices` - Array of 1D integer tensors, one for each trailing dimension
///
/// # Returns
///
/// 1D tensor with shape `[batch]` containing the gathered values
pub fn batch_gather(
    weights: &Tensor<WgpuBackend, 3>, // [batch, dim1, dim2, ...]
    indices: &[Tensor<WgpuBackend, 1, burn::tensor::Int>],
) -> Tensor<WgpuBackend, 1> {
    let n_indices = indices.len();
    if n_indices == 0 {
        // No indices means we want the entire batch flattened
        let dims = weights.dims();
        let batch_size = dims[0];
        let total_trailing: usize = dims[1..].iter().product();
        let total_size = batch_size * total_trailing;
        return weights.clone().reshape([total_size as i32]);
    }

    let dims = weights.dims();
    let batch_size = dims[0];
    let trailing_dims = &dims[1..];

    // Verify we have the right number of indices
    if indices.len() != trailing_dims.len() {
        panic!(
            "batch_gather: Expected {} index tensors (one per trailing dimension), got {}",
            trailing_dims.len(),
            indices.len()
        );
    }

    // Verify all indices have the same length (batch_size)
    for (i, idx) in indices.iter().enumerate() {
        if idx.dims()[0] != batch_size {
            panic!(
                "batch_gather: Index tensor {} has length {}, expected {} (batch_size)",
                i,
                idx.dims()[0],
                batch_size
            );
        }
    }

    let device = weights.device();

    // Compute strides for trailing dimensions
    // stride[i] = product of all dimensions after i (for indexing into trailing dims)
    let mut strides = Vec::new();
    let mut stride = 1;
    for &dim in trailing_dims.iter().rev() {
        strides.push(stride);
        stride *= dim;
    }
    strides.reverse();

    // Batch stride is the product of all trailing dimensions
    let batch_stride: usize = trailing_dims.iter().product();

    // Compute linear indices: batch_idx * batch_stride + idx0 * stride0 + idx1 * stride1 + ...
    let batch_indices: Tensor<WgpuBackend, 1, burn::tensor::Int> = Tensor::from_data(
        (0..batch_size)
            .map(|i| i as i32)
            .collect::<Vec<_>>()
            .as_slice(),
        &device,
    );

    let batch_stride_tensor =
        Tensor::from_data(vec![batch_stride as i32; batch_size].as_slice(), &device);
    let mut linear_idx = batch_indices * batch_stride_tensor;

    for (idx, &stride_val) in indices.iter().zip(strides.iter()) {
        let stride_tensor =
            Tensor::from_data(vec![stride_val as i32; batch_size].as_slice(), &device);
        linear_idx = linear_idx + idx.clone() * stride_tensor;
    }

    // Flatten weights to 1D for efficient indexing
    let total_size: usize = dims.iter().product();
    let weights_flat = weights.clone().reshape([total_size as i32]);

    // Select using linear indices (select along dimension 0 on the flattened tensor)
    weights_flat.select(0, linear_idx)
}

/// Batch gather with an extra "k" dimension for interactions.
///
/// This is similar to `batch_gather` but handles an extra trailing dimension
/// in the weights tensor. Used for categorical Gibbs sampling.
///
/// # Arguments
///
/// * `weights` - Tensor with shape [batch, k, dim1, dim2, ...]
/// * `indices` - Array of 1D integer tensors, one for each trailing dimension after k
///
/// # Returns
///
/// Tensor with shape [batch, k] containing the gathered values
pub fn batch_gather_with_k(
    weights: &Tensor<WgpuBackend, 3>,
    indices: &[Tensor<WgpuBackend, 1, burn::tensor::Int>],
) -> Tensor<WgpuBackend, 2> {
    let dims = weights.dims();
    let batch_size = dims[0];
    let k = dims[1];

    if indices.is_empty() {
        // No categorical indices, just return the tensor as-is
        return weights.clone().reshape([batch_size as i32, k as i32]);
    }

    let _device = weights.device();

    // Expand indices to include k dimension
    // For each index tensor of shape [batch], expand to [batch * k]
    let expanded_indices: Vec<Tensor<WgpuBackend, 1, burn::tensor::Int>> = indices
        .iter()
        .map(|idx| {
            // Repeat each element k times: [a, b, c] -> [a, a, ..., b, b, ..., c, c, ...]
            let idx_expanded: Tensor<WgpuBackend, 2, burn::tensor::Int> =
                idx.clone().unsqueeze_dim::<2>(1).repeat_dim(1, k);
            // Flatten to [batch * k]
            idx_expanded.reshape([(batch_size * k) as i32])
        })
        .collect();

    // Reshape weights to [batch * k, dim1, dim2, ...]
    let trailing_dims = &dims[2..];
    let flattened_batch = batch_size * k;

    // Create a 3D tensor for batch_gather
    // Reshape to [batch * k, trailing_dims...]
    let weights_reshaped = if trailing_dims.is_empty() {
        weights.clone().reshape([flattened_batch as i32, 1, 1])
    } else if trailing_dims.len() == 1 {
        weights
            .clone()
            .reshape([flattened_batch as i32, trailing_dims[0] as i32, 1])
    } else {
        weights.clone().reshape([
            flattened_batch as i32,
            trailing_dims[0] as i32,
            trailing_dims[1] as i32,
        ])
    };

    // Use batch_gather on the reshaped tensor
    let gathered = batch_gather(&weights_reshaped, &expanded_indices);

    // Reshape result from [batch * k] to [batch, k]
    gathered.reshape([batch_size as i32, k as i32])
}

/// Separate spin vs categorical states into typed tensors.
///
/// Spin states are converted to boolean tensors, categorical states to integer tensors.
pub fn split_states(states: &[Tensor<WgpuBackend, 2>], n_spin: usize) -> (SpinStates, CatStates) {
    let states_spin: SpinStates = states[..n_spin].iter().map(|s| s.clone().bool()).collect();

    let states_cat: CatStates = states[n_spin..].iter().map(|s| s.clone().int()).collect();

    (states_spin, states_cat)
}

/// A factor that defines an energy function for discrete EBMs
pub struct DiscreteEBMFactor {
    pub spin_node_groups: Vec<Block>,
    pub categorical_node_groups: Vec<Block>,
    pub weights: Tensor<WgpuBackend, 3>,
}

impl DiscreteEBMFactor {
    pub fn new(
        spin_node_groups: Vec<Block>,
        categorical_node_groups: Vec<Block>,
        weights: Tensor<WgpuBackend, 3>,
    ) -> Result<Self, String> {
        // Validate that all node groups have the same length
        let n_nodes =
            if let Some(first) = spin_node_groups.first().or(categorical_node_groups.first()) {
                first.len()
            } else {
                return Err("At least one node group must be provided".to_string());
            };

        for group in spin_node_groups
            .iter()
            .chain(categorical_node_groups.iter())
        {
            if group.len() != n_nodes {
                return Err(
                    "Every block in node_groups must contain the same number of nodes".to_string(),
                );
            }
        }

        // Validate weights shape
        let weight_dims = weights.dims();
        if weight_dims[0] != n_nodes {
            return Err("The leading dimension of weights must have the same length as the number of nodes in each node group".to_string());
        }

        if weight_dims.len() != 1 + categorical_node_groups.len() {
            return Err("The shape of the weight tensor must be [b, x_1, ..., x_k], where k is the length of categorical_node_groups".to_string());
        }

        Ok(DiscreteEBMFactor {
            spin_node_groups,
            categorical_node_groups,
            weights,
        })
    }

    /// Get all node groups (spin + categorical)
    pub fn node_groups(&self) -> Vec<Block> {
        let mut groups = self.spin_node_groups.clone();
        groups.extend(self.categorical_node_groups.clone());
        groups
    }
}

impl AbstractFactor for DiscreteEBMFactor {
    fn node_groups(&self) -> &[Block] {
        // This is a bit awkward since we need to return a slice but have two vecs
        // For now, just return spin groups (this is primarily used for validation)
        &self.spin_node_groups
    }

    fn to_interaction_groups(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<FactorInteractionGroup> {
        let mut interaction_groups = Vec::new();

        let n_spin = self.spin_node_groups.len();
        let n_cat = self.categorical_node_groups.len();
        let n_total = n_spin + n_cat;

        // Handle the interaction groups with spin head nodes
        if n_spin > 0 {
            // Generate combinations: (head_index, [tail_indices])
            // Each spin group takes a turn being the head, others are tail
            let spin_inds: Vec<usize> = (0..n_spin).collect();
            let spin_combos: Vec<(usize, Vec<usize>)> = spin_inds
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let tail: Vec<usize> = spin_inds
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, &v)| v)
                        .collect();
                    (x, tail)
                })
                .collect();

            // Collect all head nodes and tail nodes across all combos
            let mut all_head_nodes: Vec<Node> = Vec::new();
            let mut all_tail_nodes: Vec<Vec<Node>> = vec![Vec::new(); n_total - 1];

            for (head_idx, tail_inds) in &spin_combos {
                // Add head nodes from this spin group
                all_head_nodes.extend(self.spin_node_groups[*head_idx].nodes().iter().cloned());

                // Add tail nodes from other spin groups
                for (i, &tail_ind) in tail_inds.iter().enumerate() {
                    all_tail_nodes[i]
                        .extend(self.spin_node_groups[tail_ind].nodes().iter().cloned());
                }

                // Add all categorical groups as tail nodes
                for (j, cat_group) in self.categorical_node_groups.iter().enumerate() {
                    all_tail_nodes[n_spin - 1 + j].extend(cat_group.nodes().iter().cloned());
                }
            }

            // Tile the weights: repeat n_spin times along the batch dimension
            let weight_dims = self.weights.dims();
            let batch_size = weight_dims[0];
            let _new_batch_size = batch_size * n_spin;

            // Create tiled weights by repeating the tensor
            let mut tiled_weights_vec = Vec::new();
            for _ in 0..n_spin {
                tiled_weights_vec.push(self.weights.clone());
            }
            let rep_weights = Tensor::cat(tiled_weights_vec, 0);

            // Create the interaction group
            let head_block = Block::new(all_head_nodes)
                .expect("Failed to create head block for spin interaction");
            let tail_blocks: Vec<Block> = all_tail_nodes
                .into_iter()
                .map(|nodes| Block::new(nodes).expect("Failed to create tail block"))
                .collect();

            let interaction = DiscreteEBMInteraction::new(n_spin - 1, rep_weights);

            if let Ok(group) = FactorInteractionGroup::new(interaction, head_block, tail_blocks) {
                interaction_groups.push(group);
            }
        }

        // Handle the interaction groups with categorical head nodes
        if n_cat > 0 {
            let cat_inds: Vec<usize> = (0..n_cat).collect();

            // Generate combinations for categorical variables
            let cat_combos: Vec<(usize, Vec<usize>)> = cat_inds
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let tail: Vec<usize> = cat_inds
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, &v)| v)
                        .collect();
                    (x, tail)
                })
                .collect();

            for (head_idx, tail_inds) in cat_combos {
                let head_nodes = self.categorical_node_groups[head_idx].clone();

                // Tail nodes: all spin groups + selected categorical groups
                let mut tail_blocks: Vec<Block> = self.spin_node_groups.clone();
                for &i in &tail_inds {
                    tail_blocks.push(self.categorical_node_groups[i].clone());
                }

                // Reorder weight axes: move the head category dimension to position 1
                // Original: [batch, cat0, cat1, ...]
                // We want: [batch, cat_head, cat_tail0, cat_tail1, ...]
                //
                // In Python: reind = (0, combo[0] + 1, *[x + 1 for x in combo[1]])
                //            weights_reind = jnp.moveaxis(self.weights, reind, list(range(len(reind))))
                //
                // For now, if there's only one categorical variable, no reordering needed
                let weights_reind = if n_cat == 1 {
                    self.weights.clone()
                } else {
                    // Implement axis permutation
                    // This requires permute/transpose operations
                    // For 3D tensors with categorical groups, we need to reorder
                    self.permute_weights_for_categorical(head_idx, &tail_inds, device)
                };

                let interaction = DiscreteEBMInteraction::new(n_spin, weights_reind);

                if let Ok(group) = FactorInteractionGroup::new(interaction, head_nodes, tail_blocks)
                {
                    interaction_groups.push(group);
                }
            }
        }

        interaction_groups
    }
}

impl DiscreteEBMFactor {
    /// Permute weights for categorical head node processing.
    ///
    /// This implements the equivalent of jnp.moveaxis to reorder the categorical
    /// dimensions so the head category is in the right position.
    fn permute_weights_for_categorical(
        &self,
        head_idx: usize,
        _tail_inds: &[usize],
        _device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 3> {
        // Build the permutation indices
        // reind = (0, head_idx + 1, *[x + 1 for x in tail_inds])
        // This moves axis `head_idx + 1` to position 1

        let weight_dims = self.weights.dims();
        let n_dims = weight_dims.len();

        // For 3D tensor: dims are [batch, cat0, cat1]
        // If head_idx = 1, we want [batch, cat1, cat0] -> swap_dims(1, 2)

        if n_dims == 3 {
            // Only two categorical dimensions, simple swap if needed
            if head_idx == 0 {
                // Head is already at position 1, no change needed
                self.weights.clone()
            } else {
                // Swap dimensions 1 and 2
                self.weights.clone().swap_dims(1, 2)
            }
        } else {
            // For higher dimensional cases, we'd need more complex permutation
            // For now, just return the weights as-is (this is a simplification)
            self.weights.clone()
        }
    }
}

impl EBMFactor for DiscreteEBMFactor {
    fn factor_energy(
        &self,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        // Get spin values from global state
        let spin_vals = from_global_state(global_state, block_spec, &self.spin_node_groups, device);

        // Get categorical values from global state
        let cat_vals = from_global_state(
            global_state,
            block_spec,
            &self.categorical_node_groups,
            device,
        );

        // Compute spin product
        let spin_vals_bool: Vec<Tensor<WgpuBackend, 1, burn::tensor::Bool>> =
            spin_vals.iter().map(|t| t.clone().bool()).collect();
        let spin_prod = spin_product(&spin_vals_bool, device);

        // Convert categorical values to Int for batch_gather
        let cat_vals_int: Vec<Tensor<WgpuBackend, 1, burn::tensor::Int>> =
            cat_vals.iter().map(|t| t.clone().int()).collect();

        // Index into weights using categorical values
        let weights = if cat_vals_int.is_empty() {
            // No categorical variables, weights are just the batch dimension
            let dims = self.weights.dims();
            self.weights.clone().reshape([dims[0] as i32])
        } else {
            batch_gather(&self.weights, &cat_vals_int)
        };

        // Energy = -sum(weights * spin_prod)
        let energy = -(weights * spin_prod).sum();

        // Return as 1D tensor with single element
        energy.unsqueeze_dim(0)
    }
}

// ============================================================================
// Specialized Factor Types
// ============================================================================

/// A DiscreteEBMFactor that involves only spin variables.
///
/// This is a convenience wrapper around DiscreteEBMFactor with no categorical node groups.
pub struct SpinEBMFactor {
    inner: DiscreteEBMFactor,
}

impl SpinEBMFactor {
    pub fn new(node_groups: Vec<Block>, weights: Tensor<WgpuBackend, 3>) -> Result<Self, String> {
        let inner = DiscreteEBMFactor::new(node_groups, vec![], weights)?;
        Ok(SpinEBMFactor { inner })
    }

    pub fn inner(&self) -> &DiscreteEBMFactor {
        &self.inner
    }
}

impl AbstractFactor for SpinEBMFactor {
    fn node_groups(&self) -> &[Block] {
        // SpinEBMFactor only has spin node groups
        &self.inner.spin_node_groups
    }

    fn to_interaction_groups(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<FactorInteractionGroup> {
        self.inner.to_interaction_groups(device)
    }
}

impl EBMFactor for SpinEBMFactor {
    fn factor_energy(
        &self,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        self.inner.factor_energy(global_state, block_spec, device)
    }
}

/// A DiscreteEBMFactor that involves only categorical variables.
///
/// This is a convenience wrapper around DiscreteEBMFactor with no spin node groups.
pub struct CategoricalEBMFactor {
    inner: DiscreteEBMFactor,
}

impl CategoricalEBMFactor {
    pub fn new(node_groups: Vec<Block>, weights: Tensor<WgpuBackend, 3>) -> Result<Self, String> {
        let inner = DiscreteEBMFactor::new(vec![], node_groups, weights)?;
        Ok(CategoricalEBMFactor { inner })
    }

    pub fn inner(&self) -> &DiscreteEBMFactor {
        &self.inner
    }
}

impl AbstractFactor for CategoricalEBMFactor {
    fn node_groups(&self) -> &[Block] {
        // CategoricalEBMFactor only has categorical node groups
        &self.inner.categorical_node_groups
    }

    fn to_interaction_groups(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<FactorInteractionGroup> {
        self.inner.to_interaction_groups(device)
    }
}

impl EBMFactor for CategoricalEBMFactor {
    fn factor_energy(
        &self,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        self.inner.factor_energy(global_state, block_spec, device)
    }
}

/// A discrete factor with a square interaction weight tensor.
///
/// If a discrete factor is square (shape [b, x, x, ..., x]), the interaction groups
/// corresponding to different choices of the head node blocks can be merged for
/// improved runtime performance.
pub struct SquareDiscreteEBMFactor {
    inner: DiscreteEBMFactor,
}

impl SquareDiscreteEBMFactor {
    pub fn new(
        spin_node_groups: Vec<Block>,
        categorical_node_groups: Vec<Block>,
        weights: Tensor<WgpuBackend, 3>,
    ) -> Result<Self, String> {
        // Validate that weights are square (all non-batch dimensions equal)
        let weight_dims = weights.dims();
        if weight_dims.len() > 2 {
            let target_shape = weight_dims[1];
            for &dim in &weight_dims[1..] {
                if dim != target_shape {
                    return Err("Interaction tensor is not square".to_string());
                }
            }
        }

        let inner = DiscreteEBMFactor::new(spin_node_groups, categorical_node_groups, weights)?;
        Ok(SquareDiscreteEBMFactor { inner })
    }

    pub fn inner(&self) -> &DiscreteEBMFactor {
        &self.inner
    }
}

impl AbstractFactor for SquareDiscreteEBMFactor {
    fn node_groups(&self) -> &[Block] {
        // Return spin node groups (for validation purposes)
        &self.inner.spin_node_groups
    }

    fn to_interaction_groups(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<FactorInteractionGroup> {
        // Get base interaction groups

        // For square factors, we could merge groups here
        // For now, just return the base groups
        // TODO: Implement group merging for optimization
        self.inner.to_interaction_groups(device)
    }
}

impl EBMFactor for SquareDiscreteEBMFactor {
    fn factor_energy(
        &self,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        self.inner.factor_energy(global_state, block_spec, device)
    }
}

/// A DiscreteEBMFactor with only categorical variables and a square weight tensor.
pub struct SquareCategoricalEBMFactor {
    inner: SquareDiscreteEBMFactor,
}

impl SquareCategoricalEBMFactor {
    pub fn new(node_groups: Vec<Block>, weights: Tensor<WgpuBackend, 3>) -> Result<Self, String> {
        let inner = SquareDiscreteEBMFactor::new(vec![], node_groups, weights)?;
        Ok(SquareCategoricalEBMFactor { inner })
    }
}

impl AbstractFactor for SquareCategoricalEBMFactor {
    fn node_groups(&self) -> &[Block] {
        // SquareCategoricalEBMFactor only has categorical node groups
        &self.inner.inner.categorical_node_groups
    }

    fn to_interaction_groups(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<FactorInteractionGroup> {
        self.inner.to_interaction_groups(device)
    }
}

impl EBMFactor for SquareCategoricalEBMFactor {
    fn factor_energy(
        &self,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        self.inner.factor_energy(global_state, block_spec, device)
    }
}
