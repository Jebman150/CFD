#pragma once

#include "../engine/grid.hpp"
#include "parameter.hpp"
#include <SFML/Graphics.hpp>

class Renderer {
    sf::RenderWindow window;

    GraphicalParameter parameter;

    void drawPressureField(const Eigen::MatrixXd& gridValues);
public:
    void initialize();
    void updateFrame(const Grid& grid);
    
    bool active() { return window.isOpen(); }
};