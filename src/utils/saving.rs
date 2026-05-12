use std::path::Path;

use serde_json::Value;

use crate::primitives::Module;

pub fn save_module(module: &dyn Module, path: &Path) -> anyhow::Result<()> {
    let state = module.dump_state();
    let json = serde_json::to_string_pretty(&state)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn load_state_from_file(path: &Path) -> anyhow::Result<Value> {
    let json = std::fs::read_to_string(path)?;
    let state: Value = serde_json::from_str(&json)?;
    Ok(state)
}
