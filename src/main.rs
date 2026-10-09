use crate::app::Application;

mod app;
mod gui;
mod simulation;
mod rendering;
mod logging;

#[macroquad::main("CFD")]
async fn main() {
    println!("------- Initialization -------");
    let mut app = Application::new();

    println!("------- Main Loop -------");
    match app.run().await {
        Ok(()) => println!("Successfully closed"),
        Err(()) => println!("An error occurred"),
    }


    println!("------- Program closed -------");


    /*let mut simulation = initialize();
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
    cleanup(&simulation);*/
}

/*
mod engine;
mod rendering;

use macroquad::prelude::*;

use macroquad::time::draw_fps;

use std::fmt::Result;
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

impl Simulation {
    pub fn new() -> Self {
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
}

fn simulation_step(state: &mut engine::state::State2D, engine: &Engine) {
    
}

fn cleanup(_simulation: &Simulation) {

}

enum AppState {
    PAUSED,
    RUNNING
}

struct App {
    pub app_state: AppState,
    pub simulation: Simulation,
    //pub gui: GUI,

    pub is_running: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            app_state: AppState::PAUSED,
            simulation: Simulation::new(),

        }
    }

    pub async fn run(&mut self) -> Result {
        while self.is_running {
            match self.app_state {
                AppState::PAUSED => self.handle_gui(),
                AppState::RUNNING => {self.handle_gui(); self.simulation_step()},
            }
            simulation_step(&mut simulation.state, &simulation.engine);
            //break;

            simulation.renderer.update(&simulation.state, &simulation.engine.grid);

            //thread::sleep(Duration::from_secs(1));

            draw_fps();
            next_frame().await;
        }
        Ok(())
    }

    fn handle_gui(&mut self) {

    }

    fn simulation_step(&mut self) {
        self.simulation.compute_step();
        
        engine.apply_boundary_condition(state);

        engine.advect_velocities(state);

        engine.spawn_smoke(state);
        engine.advect(&state.velocity, &mut state.smoke);

        engine.apply_boundary_condition(state);

        engine.compute_pressure(state);

        engine.apply_boundary_condition(state);
    }
} */

