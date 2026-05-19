#include "engine/engine.hpp"
#include "graphics/renderer.hpp"
#include "engine/tracer.hpp"

#include <chrono>
#include <thread>
#include <iostream>

using namespace std::chrono_literals;

int main() {
    Tracer tracer;
    engine::Engine engine;
    renderer::Renderer renderer;

    engine.initSim();
    renderer.initialize();
    renderer.setVisMode(renderer::VisMode::Smoke);
    engine.spawnSmoke();

    while(renderer.active()) {
        tracer.startJob("Frame");
        auto frameStart = std::chrono::high_resolution_clock::now();
        engine.adjustTimestep();
        //std::this_thread::sleep_for(1000ms / 30 * engine.getDeltatime());


        tracer.startJob("Advect");
        engine.applyBoundaryCondition();
        engine.advect();
        tracer.endJob("Advect");

        tracer.startJob("Project");
        engine.applyBoundaryCondition();
        engine.project();
        tracer.endJob("Project");

        tracer.startJob("Render");
        renderer.updateFrame(engine.getGrid());
        tracer.endJob("Render");
        tracer.endJob("Frame");
        auto frameEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> deltaTime = frameEnd - frameStart;
        std::cout << "FPS: " << 1/(deltaTime.count() / 1000.f) << std::endl;
    }
    std::cout << "Average job times: " << std::endl
        << " | Advect: " << tracer.getAvgTime("Advect") << std::endl
        << " | Project: " << tracer.getAvgTime("Project") << std::endl
        << " | Render: " << tracer.getAvgTime("Render") << std::endl
        << " | Total: " << tracer.getAvgTime("Frame") << std::endl;

    return 0;
}