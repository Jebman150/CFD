#pragma once

#include "../engine/grid.hpp"
#include "parameter.hpp"
#include <SFML/Graphics.hpp>

namespace renderer {

using engine::Grid;

enum VisMode {
    Pressure,
    Speed,
    Divergence,
    Smoke,
    NONE
};

enum VectorFieldMode {
    VelocityHR,
    VelocityDebug,
    OFF
};

class Renderer {
    sf::RenderWindow window;

    GraphicalParameter parameter;

    VisMode visMode = NONE;
    VectorFieldMode vecMode = OFF;
    sf::Vector2i resolution = {100, 100};

    sf::Vector2i gridSize;
    sf::Vector2f cellSize;

    void drawCoordinateSystem();
    void drawCells(const Grid& grid);
    void drawFaces(const Grid& grid);

    void drawDivergenceField(const Grid& grid);

    void drawVelocityPlaneDebug(const Grid& grid);

    void drawVelocityFieldHR(const Grid& grid, sf::Vector2i res);
public:
    void initialize();
    void updateFrame(const Grid& grid);
    void setVisMode(VisMode mode) { visMode = mode; }
    void setVecMode(VectorFieldMode mode) { vecMode = mode; }
    
    bool active() { return window.isOpen(); }

private:
    template<typename Sampler>
    void drawScalarHRField(Sampler sampler, sf::Vector2i res) {
        sf::Vector2f stride = {gridSize.x / float(res.x), gridSize.y / float(res.y)};

        for(int i = 0; i < res.x; i++) {
            for(int j = 0; j < res.y; j++) {
                float val = sampler(stride.x * i, stride.y * j);
                auto position = parameter.targetRect.position + sf::Vector2f(stride.x * i * cellSize.x, stride.y * j * cellSize.y);
                sf::RectangleShape rect({stride.x * cellSize.x, stride.y * cellSize.y});
                rect.setPosition(position);
                rect.setFillColor(parameter.smokeColor.eval(val));

                window.draw(rect);
            }
        }
    }
};

}