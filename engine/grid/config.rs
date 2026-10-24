use serde::Deserialize;
use std::fs;
use anyhow::Result;

#[derive(Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ConditionType {
    Neumann,
    Dirichlet
}

#[derive(Deserialize)]
pub struct BoundaryCondition {
    pub condition_type: ConditionType,
    pub value: f32
}

#[derive(Deserialize)]
pub struct Config {
    pub grid_size: Vec<usize>,
    pub boundary_condition: Vec<BoundaryCondition>,
    pub delta_x: f32,
}

impl Config {
    pub fn from_file(path: &str) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }
}