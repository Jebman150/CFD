use tracing::instrument;
use winit::event_loop::EventLoop;

use crate::application::Application;

mod application;

#[instrument]
fn main() -> anyhow::Result<()> {
    let _stdout_subscriber = tracing_subscriber::fmt::init();
    tracing::info!("------- Initialization -------");
    let mut app = match Application::new() {
        Ok(app) => app,
        Err(err) => {
            tracing::error!("Could not initialize application. Error: {}", err);
            return Err(err);
        }
    };
    app.rebuild_scene();

    puffin::set_scopes_on(true);

    let server_addr =
        format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);

    let _puffin_server =
        puffin_http::Server::new(&server_addr)
            .expect("Failed to start Puffin server");
    tracing::info!("Opened puffin server at {}", &server_addr);

    tracing::info!("Rayon worker threads: {}", rayon::current_num_threads());

    let event_loop = EventLoop::with_user_event().build()?;
    tracing::info!("Started event loop");

    tracing::info!("------- Main Loop -------");
    event_loop.run_app(&mut app)?;


    tracing::info!("------- Program closed -------");
    Ok(())
}