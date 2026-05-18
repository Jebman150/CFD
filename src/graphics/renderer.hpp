#pragma once

#include "../engine/grid.hpp"
#include "parameter.hpp"
#include <SFML/Graphics.hpp>

class Renderer {
    sf::RenderWindow window;

    GraphicalParameter parameter;

    void drawCoordinateSystem();
    void drawCellOutline(Eigen::Vector2i dimensions);
    void drawPressureField(const Eigen::MatrixXd& gridValues);
    void drawDivergenceField(const Grid& grid);
    void drawVelocityPlane(const Grid& grid);

    void drawVelocityFieldHR(const Grid& grid, sf::Vector2i res);
    void drawPressureFieldHR(const Grid& grid, sf::Vector2i res);
public:
    void initialize();
    void updateFrame(const Grid& grid);
    
    bool active() { return window.isOpen(); }
};