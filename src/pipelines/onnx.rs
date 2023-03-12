use ndarray::{Dimension, IxDyn};
use ort::tensor::DynOrtTensor;
use std::collections::HashMap;
use tch::Tensor;

#[derive(Debug)]
pub struct ONNXLayerCache {
    values: HashMap<String, Tensor>,
}

impl ONNXLayerCache {
    pub fn from_ort_output(
        ort_output: &'_ Vec<DynOrtTensor<IxDyn>>,
        key_value_names: &HashMap<&str, usize>,
    ) -> ONNXLayerCache {
        let values = key_value_names
            .iter()
            .filter(|(name, _)| name.contains(".key") | name.contains(".value"))
            .map(|(name, pos)| {
                let value = ort_output[*pos].try_extract::<f32>().unwrap();
                (
                    name.to_string(),
                    Tensor::of_slice(value.view().as_slice().unwrap()).view(
                        value
                            .view()
                            .dim()
                            .as_array_view()
                            .iter()
                            .map(|dim| *dim as i64)
                            .collect::<Vec<_>>()
                            .as_slice(),
                    ),
                )
            })
            .collect::<HashMap<String, Tensor>>();

        ONNXLayerCache { values }
    }
}

// // WORKING IMPLEMENTATION 2
// let values = decoder_name_mapping
//     .decoder_key_value_output_names
//     .iter()
//     .filter(|(name, pos)| name.contains(".key") | name.contains(".value"))
//     .map(|(name, pos)| {
//         (
//             name.to_string(),
//             array![outputs[*pos].try_extract::<f32>().unwrap()],
//         )
//     })
//     .collect::<HashMap<String, ArrayBase<OwnedRepr<OrtOwnedTensor<f32, IxDyn>>, Ix1>>>();
