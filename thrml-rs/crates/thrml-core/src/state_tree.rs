use crate::backend::WgpuBackend;
use crate::blockspec::BlockSpec;
use burn::tensor::Tensor;

pub trait StateLeaf: Clone {
    fn stack(leaves: &[Self]) -> Self
    where
        Self: Sized;
    fn take_indices(&self, indices: Tensor<WgpuBackend, 1, burn::tensor::Int>) -> Self;
}

impl<const D: usize> StateLeaf for Tensor<WgpuBackend, D> {
    fn stack(leaves: &[Self]) -> Self {
        if leaves.is_empty() {
            panic!("Cannot stack empty leaves");
        }
        if leaves.len() == 1 {
            return leaves[0].clone();
        }
        // Use Burn's concat operation
        Tensor::cat(leaves.to_vec(), 0)
    }

    fn take_indices(&self, indices: Tensor<WgpuBackend, 1, burn::tensor::Int>) -> Self {
        // Burn's select operation for indexing
        // select takes (dim, indices) where indices is a 1D Int tensor
        // select takes ownership, so we clone first
        self.clone().select(0, indices)
    }
}

pub fn block_state_to_global<L: StateLeaf>(block_state: &[L], spec: &BlockSpec) -> Vec<L> {
    let mut global_state = Vec::new();
    for sd_indexes in &spec.block_to_global_slice_spec {
        if sd_indexes.is_empty() {
            continue; // Skip None equivalent
        }
        let collected: Vec<L> = sd_indexes.iter().map(|&i| block_state[i].clone()).collect();
        if collected.len() == 1 {
            global_state.push(collected[0].clone());
        } else {
            global_state.push(L::stack(&collected));
        }
    }
    global_state
}

pub fn from_global_state<L: StateLeaf>(
    global_state: &[L],
    spec_from: &BlockSpec,
    blocks_to_extract: &[crate::block::Block],
    device: &burn::backend::wgpu::WgpuDevice,
) -> Vec<L> {
    let mut result = Vec::new();
    for block in blocks_to_extract {
        let (sd_ind, slices) = spec_from
            .get_node_locations(block)
            .expect("Failed to get node locations");
        // Convert slices to tensor indices (as Int tensor)
        // Use from_data with proper type annotation for Int tensor
        let indices: Tensor<WgpuBackend, 1, burn::tensor::Int> = Tensor::from_data(
            slices
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        );
        let extracted = global_state[sd_ind].take_indices(indices);
        result.push(extracted);
    }
    result
}

pub fn make_empty_block_state(
    blocks: &[crate::block::Block],
    node_shape_dtypes: &indexmap::IndexMap<crate::node::NodeType, crate::node::TensorSpec>,
    batch_shape: Option<&[usize]>,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Vec<Tensor<WgpuBackend, 1>> {
    let mut state = Vec::new();
    for block in blocks {
        let _spec = node_shape_dtypes
            .get(block.node_type())
            .expect("Node type not found in node_shape_dtypes");
        let block_len = block.len();
        let shape: [usize; 1] = if let Some(batch) = batch_shape {
            let total = batch.iter().product::<usize>() * block_len;
            [total]
        } else {
            [block_len]
        };
        // Create zero tensor - dtype will be determined by usage context
        // For now, use f32 as default (will be cast as needed)
        let tensor = Tensor::zeros(shape, device);
        state.push(tensor);
    }
    state
}
