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

    void drawCoordinateSystem();
    void drawCells(const Grid& grid);
    void drawFaces(const Grid& grid);

    void drawDivergenceField(const Grid& grid);

    void drawVelocityPlaneDebug(const Grid& grid);

    void drawVelocityFieldHR(const Grid& grid, sf::Vector2i res);

    void drawPressureFieldHR(const Grid& grid, sf::Vector2i res);
    void drawSpeedFieldHR(const Grid& grid, sf::Vector2i res);
    void drawSmokeFieldHR(const Grid& grid, sf::Vector2i res);
public:
    void initialize();
    void updateFrame(const Grid& grid);
    void setVisMode(VisMode mode) { visMode = mode; }
    
    bool active() { return window.isOpen(); }
};

}