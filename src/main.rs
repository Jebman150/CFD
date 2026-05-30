mod engine;
mod rendering;
use macroquad::prelude::*;

use macroquad::time::draw_fps;

use std::thread;
use std::time::Duration;

use rendering::Renderer;
use engine::Engine;

struct Simulation {
    pub state: engine::state::State2D,
    pub engine: engine::Engine,
    pub renderer: rendering::Renderer,
    pub is_running: bool,
}

fn initialize() -> Simulation {
    let engine: Engine = Engine::new("config/");
    println!(" Read engine config from directory config/");

    let initial_state = engine::state::State2D::new(&engine.grid);
    println!(" Created initial state");

    let renderer: Renderer = Renderer::new();
    println!(" Created renderer");

    Simulation {
        state: initial_state,
        engine: engine,
        renderer: renderer,
        is_running: true,
    }
}

fn simulation_step(state: &mut engine::state::State2D, engine: &Engine) {
    engine.apply_boundary_condition(state);

    engine.advect_velocities(state);

    engine.spawn_smoke(state);
    engine.advect(&state.velocity, &mut state.smoke);

    engine.apply_boundary_condition(state);

    engine.compute_pressure(state);

    engine.apply_boundary_condition(state);
}

fn cleanup(_simulation: &Simulation) {

}

#[macroquad::main("CFD")]
async fn main() {
    println!("------- Initialization -------");
    let mut simulation = initialize();
    simulation.engine.initialize_grid(&mut simulation.state);

    println!("------- Main Loop -------");
    while simulation.is_running {
        simulation_step(&mut simulation.state, &simulation.engine);
        //break;

        simulation.renderer.update(&simulation.state, &simulation.engine.grid);

        //thread::sleep(Duration::from_secs(1));

        draw_fps();
        next_frame().await;
    }

    println!("------- Cleaning up -------");
    cleanup(&simulation);
}