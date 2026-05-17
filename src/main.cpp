#include "engine/engine.hpp"
#include "graphics/renderer.hpp"

#include <chrono>
#include <thread>

using namespace std::chrono_literals;

int main() {
    Engine engine;
    Renderer renderer;

    engine.initSim();
    renderer.initialize();

    while(renderer.active()) {
        std::this_thread::sleep_for(1000ms / 60);

        engine.advect();
        engine.applyBoundaryCondition();

        engine.applyBoundaryCondition();
        engine.project();
        renderer.updateFrame(engine.getGrid());
    }

    return 0;
}