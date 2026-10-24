use tracing::{Level, instrument};
use winit::event_loop::EventLoop;

use crate::app::Application;

mod app;
mod gui;
mod simulation;
mod rendering;
mod logging;

#[instrument]
fn main() -> anyhow::Result<()> {
    let _stdout_subscriber = tracing_subscriber::fmt::init();
    tracing::event!(Level::INFO, "------- Initialization -------");
    let mut app = match Application::new() {
        Ok(app) => app,
        Err(err) => {
            tracing::event!(Level::ERROR, "Could not initialize application. Error: {}", err);
            return Err(err);
        }
    };
    app.rebuild_scene();
    logging::clear_log_directory("logs")?;

    puffin::set_scopes_on(true);

    let server_addr =
        format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);

    let _puffin_server =
        puffin_http::Server::new(&server_addr)
            .expect("Failed to start Puffin server");
    tracing::info!("Opened puffin server at {}", &server_addr);

    tracing::event!(Level::INFO, "Rayon worker threads: {}", rayon::current_num_threads());

    let event_loop = EventLoop::with_user_event().build()?;
    tracing::event!(Level::INFO, "Started event loop");

    tracing::event!(Level::INFO, "------- Main Loop -------");
    event_loop.run_app(&mut app)?;


    tracing::event!(Level::INFO, "------- Program closed -------");
    Ok(())
}