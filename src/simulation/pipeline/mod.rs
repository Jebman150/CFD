use std::time::Instant;

use colored::*;
pub mod operator;

pub struct SimulationPipeline {
    operations: Vec<Box<dyn operator::SimulationOperator>>,
}

impl SimulationPipeline {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn add_step(&mut self, op: Box<dyn operator::SimulationOperator>) -> &mut Self {
        self.operations.push(op);
        self
    }

    pub fn run(&mut self, data: &mut super::data::SimulationData , context: &mut super::data::grid::Grid2D) -> Result<u128, String> {
        let start = Instant::now();
        for op in &mut self.operations {
            match op.execute(data, context) {
                Ok(time) => println!(" | Executed {} in {} micros", op.get_name(), time),
                Err(msg) => {
                    println!("{}", format!("ERROR: {} failed with msg {}", op.get_name(), msg).red());
                    return Err(format!("{} failed with msg {}", op.get_name(), msg))
                },
            }
        }
        Ok(start.elapsed().as_micros())
    }
}