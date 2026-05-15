#include "engine/engine.hpp"
#include "graphics/renderer.hpp"

int main() {
    Engine engine;
    Renderer renderer;

    engine.initSim();
    renderer.initialize();

    while(renderer.active()) {
        renderer.updateFrame(engine.getGrid());
    }

    return 0;
}