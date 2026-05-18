#include "engine/engine.hpp"
#include "graphics/renderer.hpp"
#include "engine/tracer.hpp"

#include <chrono>
#include <thread>
#include <iostream>

using namespace std::chrono_literals;

int main() {
    Tracer tracer;
    Engine engine;
    Renderer renderer;

    engine.initSim();
    renderer.initialize();

    while(renderer.active()) {
        engine.adjustTimestep();
        std::this_thread::sleep_for(1000ms / 30 * engine.getDeltatime());

        tracer.startJob("Advect");
        engine.applyBoundaryCondition();
        engine.advect();
        tracer.endJob("Advect");

        tracer.startJob("Project");
        engine.applyBoundaryCondition();
        engine.project();
        tracer.endJob("Project");
        renderer.updateFrame(engine.getGrid());
    }
    std::cout << "Average job times: " << std::endl
        << " | Advect: " << tracer.getAvgTime("Advect") << std::endl
        << " | Project: " << tracer.getAvgTime("Project") << std::endl;

    return 0;
}