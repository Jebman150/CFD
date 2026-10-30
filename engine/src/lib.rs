mod engine;
mod state;
mod gpu;

pub use engine::SimulationEngine;
pub use state::SimulationState;
pub use gpu::GPUContext;
pub use state::fields::SIUnitDimensions;
pub use state::ScalarFieldID;
pub use state::VectorFieldID;
pub use state::parameter_registry::ParamID;
pub use state::fields::VectorFieldStorage;
pub use gpu::kernel::ParameterLayout;

pub use engine::operations::divergence::DivergenceOperator;
pub use engine::operations::advection::ScalarAdvectionOperator;