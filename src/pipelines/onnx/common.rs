use ort::Session;
use std::collections::HashMap;

#[derive(Debug)]
pub(crate) struct InputOutputNameMapping {
    pub(crate) input_names: Vec<String>,
    pub(crate) output_names: HashMap<String, usize>,
    pub(crate) key_value_output_names: HashMap<String, usize>,
}

pub(crate) fn get_input_output_mapping(session: &Session) -> InputOutputNameMapping {
    let input_names = session
        .inputs
        .iter()
        .map(|input| input.name.clone())
        .collect::<Vec<String>>();

    let output_names = session
        .outputs
        .iter()
        .enumerate()
        .map(|(pos, output)| (output.name.clone(), pos))
        .collect::<HashMap<String, usize>>();

    let mut key_value_output_names = output_names
        .iter()
        .filter(|(name, _)| name.contains(".key") | name.contains(".value"))
        .map(|(name, pos)| (name.clone(), *pos))
        .collect::<HashMap<String, usize>>();

    if key_value_output_names.is_empty() {
        key_value_output_names = output_names
            .iter()
            .filter(|(name, _)| name.contains("key_value"))
            .map(|(name, pos)| (name.clone(), *pos))
            .collect::<HashMap<String, usize>>();
    }

    InputOutputNameMapping {
        input_names,
        output_names,
        key_value_output_names,
    }
}
