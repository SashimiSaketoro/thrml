use crate::backend::WgpuBackend;
use crate::block::Block;
use burn::tensor::Tensor;

/// Defines computational dependencies for conditional sampling updates.
///
/// An `InteractionGroup` specifies information that is required to update the state of some subset
/// of the nodes of a PGM during a block sampling routine.
pub struct InteractionGroup {
    /// The nodes whose conditional updates should be affected by this InteractionGroup.
    pub head_nodes: Block,
    /// The nodes whose state information is required to update `head_nodes`.
    pub tail_nodes: Vec<Block>,
    /// The static information associated with the interaction (PyTree equivalent).
    /// This is a tensor representing the interaction weights/parameters.
    pub interaction: Tensor<WgpuBackend, 2>,
}

impl InteractionGroup {
    pub fn new(
        interaction: Tensor<WgpuBackend, 2>,
        head_nodes: Block,
        tail_nodes: Vec<Block>,
    ) -> Result<Self, String> {
        let interaction_size = head_nodes.len();

        for block in &tail_nodes {
            if block.len() != interaction_size {
                return Err(
                    "All tail node blocks must have the same length as head_nodes".to_string(),
                );
            }
        }

        // Verify interaction tensor has correct leading dimension
        let interaction_dims = interaction.dims();
        if interaction_dims[0] != interaction_size {
            return Err(
                "All arrays in interaction must have leading dimension equal to the length of head_nodes".to_string()
            );
        }

        Ok(InteractionGroup {
            head_nodes,
            tail_nodes,
            interaction,
        })
    }
}
