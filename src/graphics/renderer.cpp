#include "renderer.hpp"

/*
    Creates a window and prepares the renderer for drawing.
*/
void Renderer::initialize() {
    window = sf::RenderWindow{sf::VideoMode({parameter.windowSize.x, parameter.windowSize.y}), "CFD"};
}

/*
    Updates the active window to show the grid values.

    @param grid
        The grid-object to show on screen
*/
void Renderer::updateFrame(const Grid& grid) {
    window.clear(parameter.backgroundColor);

    while (const std::optional event = window.pollEvent())
    {
        if (event->is<sf::Event::Closed>())
            window.close();
    }

    auto& pressureVals = grid.getPressureValues();
    drawPressureField(pressureVals);

    window.display();
}

/*
    Draws the pressure of the given grid.

    @param gridValues
        Matrix of the pressure values
*/
void Renderer::drawPressureField(const Eigen::MatrixXd& gridValues) {
    sf::Vector2f gridSize = {gridValues.cols(), gridValues.rows()};
    sf::Vector2f cellSize = {parameter.targetRect.size.x / gridSize.x, parameter.targetRect.size.y / gridSize.y};

    for(int x = 0; x < gridSize.x; x++) {
        for(int y = 0; y < gridSize.y; y++) {
            auto rect = sf::RectangleShape(cellSize);
            float value = gridValues(x, y);
            rect.setOutlineColor(parameter.gridCellColor);
            rect.setFillColor(parameter.pressureColor.eval(value));
            rect.setOutlineThickness(2);
            rect.setPosition(parameter.targetRect.position + sf::Vector2f{x*cellSize.x, y*cellSize.y});

            window.draw(rect);
        }
    }
}