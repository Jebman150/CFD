use std::collections::HashMap;

use super::pressure_solution::PressureSolver;

#[derive(Debug)]
pub struct SolverDescriptor {
    pub id: usize,
    pub name: &'static str,

    pub create: fn() -> Box<dyn PressureSolver>,
}

pub struct SolverRegistry {
    solvers: HashMap<usize, SolverDescriptor>,
}

impl SolverRegistry {
    pub fn new() -> Self {
        Self {
            solvers: HashMap::new(),
        }
    }

    pub fn register(&mut self, solver_desc: SolverDescriptor) {
        println!("Registered solver: {:?}", solver_desc);
        self.solvers.insert(solver_desc.id, solver_desc);
    }

    pub fn create_solver(&mut self, id: usize) -> Result<Box<dyn PressureSolver>, String>{
        match self.solvers.get(&id) {
            Some(desc) => Ok((desc.create)()),
            None => Err(format!("Could not find solver with id {}", id)),
        }
    }

    pub fn list_solver(&self) -> Vec<(usize, &'static str)> {
        let mut result = Vec::new();
        for s in self.solvers.values() {
            result.push((s.id, s.name));
        }
        result
    }

    pub fn get_solver_name(&self, id: usize) -> Result<&'static str, String> {
        match self.solvers.get(&id) {
            Some(solver) => Ok(solver.name),
            None => Err(format!("Could not find solver with id {}", id)),
        }
    }
}