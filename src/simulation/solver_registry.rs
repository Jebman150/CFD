use std::collections::HashMap;

use crate::simulation::solver::Solver;

#[derive(Debug)]
pub struct SolverDescriptor {
    pub id: usize,
    pub name: &'static str,

    pub create: fn() -> Box<dyn Solver>,
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

    pub fn create_solver(&mut self, id: usize) -> Result<Box<dyn Solver>, String>{
        match self.solvers.get(&id) {
            Some(desc) => Ok((desc.create)()),
            None => Err(format!("Could not find solver with id {}", id)),
        }
    }
}