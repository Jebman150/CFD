#include "renderer.hpp"
#include<iostream>

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

    //drawPressureField(grid.getPressureValues());
    //drawDivergenceField(grid);
    //drawCellOutline({grid.getWidth(), grid.getHeight()});
    //drawXVelocityField(grid.getVelocityXValues());
    //drawYVelocityField(grid.getVelocityYValues());
    drawPressureFieldHR(grid, {100, 100});
    drawVelocityFieldHR(grid, {50, 50});

    window.display();
}

/*
    Draws the outline of the discretization grid.

    @param dimension
        Grid dimension
*/
void Renderer::drawCellOutline(Eigen::Vector2i dimension) {
    sf::Vector2f cellSize = {parameter.targetRect.size.x / float(dimension.x()), parameter.targetRect.size.y / float(dimension.y())};

    for(int x = 0; x < dimension.x(); x++) {
        for(int y = 0; y < dimension.y(); y++) {
            auto rect = sf::RectangleShape(cellSize);
            rect.setOutlineColor(parameter.gridCellColor);
            rect.setFillColor(sf::Color::Transparent);
            rect.setOutlineThickness(2);
            rect.setPosition(parameter.targetRect.position + sf::Vector2f{x*cellSize.x, y*cellSize.y});

            window.draw(rect);
        }
    }
}

/*
    Draws the pressure of the given grid.

    @param gridValues
        Matrix of the pressure values
*/
void Renderer::drawPressureField(const Eigen::MatrixXd& gridValues) {
    sf::Vector2i gridSize = {static_cast<int>(gridValues.rows()), static_cast<int>(gridValues.cols())};
    sf::Vector2f cellSize = {parameter.targetRect.size.x / float(gridSize.x), parameter.targetRect.size.y / float(gridSize.y)};

    for(int x = 0; x < gridSize.x; x++) {
        for(int y = 0; y < gridSize.y; y++) {
            auto rect = sf::RectangleShape(cellSize);
            float value = gridValues(x, y);
            rect.setFillColor(parameter.pressureColor.eval(value));
            rect.setPosition(parameter.targetRect.position + sf::Vector2f{x*cellSize.x, y*cellSize.y});

            window.draw(rect);
        }
    }
}

/*
    Draws the divergence of the given grid.

    @param grid
*/
void Renderer::drawDivergenceField(const Grid& grid) {
    sf::Vector2i gridSize = {grid.getWidth(), grid.getHeight()};
    sf::Vector2f cellSize = {parameter.targetRect.size.x / float(gridSize.x), parameter.targetRect.size.y / float(gridSize.y)};

    for(double x = 0.5; x < gridSize.x; x++) {
        for(double y = 0.5; y < gridSize.y; y++) {
            auto rect = sf::RectangleShape(cellSize);
            float value = grid.getDivergence({x, y});
            rect.setFillColor(parameter.pressureColor.eval(value));
            rect.setPosition(parameter.targetRect.position + sf::Vector2f{(x-0.5)*cellSize.x, (y-0.5)*cellSize.y});

            window.draw(rect);
        }
    }
}

/*
    Draws the given velocity field in the x direction.

    @param gridValues
        Matrix of velocity values
*/
void Renderer::drawXVelocityField(const Eigen::MatrixXd& gridValues) {
    sf::Vector2i gridSize = {static_cast<int>(gridValues.rows() - 1), static_cast<int>(gridValues.cols())};
    sf::Vector2f cellSize = {parameter.targetRect.size.x / float(gridSize.x), parameter.targetRect.size.y / float(gridSize.y)};
    sf::Vector2f offset = {0, cellSize.y / 2.f};


    for(int x = 0; x < (gridSize.x + 1); x++) {
        for(int y = 0; y < gridSize.y; y++) {
            float value = gridValues(x, y);
            auto rect = sf::RectangleShape({value * 40, 5});
            rect.setFillColor(parameter.velocityColor.eval(value));
            rect.setPosition(parameter.targetRect.position + offset + sf::Vector2f{x*cellSize.x, y*cellSize.y});

            window.draw(rect);
        }
    }
}

/*
    Draws the given velocity field in the y direction.

    @param gridValues
        Matrix of velocity values
*/
void Renderer::drawYVelocityField(const Eigen::MatrixXd& gridValues) {
    sf::Vector2i gridSize = {static_cast<int>(gridValues.rows()), static_cast<int>(gridValues.cols() - 1)};
    sf::Vector2f cellSize = {parameter.targetRect.size.x / float(gridSize.x), parameter.targetRect.size.y / float(gridSize.y)};
    sf::Vector2f offset = {cellSize.x / 2.f, 0};


    for(int x = 0; x < gridSize.x; x++) {
        for(int y = 0; y < (gridSize.y + 1); y++) {
            float value = gridValues(x, y);
            auto rect = sf::RectangleShape({5, value * 40});
            rect.setFillColor(parameter.velocityColor.eval(value));
            rect.setPosition(parameter.targetRect.position + offset + sf::Vector2f{x*cellSize.x, y*cellSize.y});

            window.draw(rect);
        }
    }
}

/*
    Draws a high resolution velocity field by bilinear interpolation of the off-grid velocities.

    @param grid
    @param res Target resolution of the field
*/
void Renderer::drawVelocityFieldHR(const Grid& grid, sf::Vector2i res) {
    sf::Vector2i gridSize = {grid.getWidth(), grid.getHeight()};
    sf::Vector2f cellSize = {parameter.targetRect.size.x / float(gridSize.x), parameter.targetRect.size.y / float(gridSize.y)};
    sf::Vector2f stride = {gridSize.x / float(res.x), gridSize.y / float(res.y)};

    for(int i = 0; i < res.x; i++) {
        for(int j = 0; j < res.y; j++) {
            auto vel = grid.interpolateVelocity({stride.x * i, stride.y * j});
            auto position = parameter.targetRect.position + sf::Vector2f(stride.x * i * cellSize.x, stride.y * j * cellSize.y);
            sf::Vertex line[] = {
                {position, sf::Color::Cyan},
                {position + sf::Vector2f(vel.x(), vel.y()) * 20.f, sf::Color::Cyan}
            };

            window.draw(line, 2, sf::PrimitiveType::Lines);
        }
    }
}

/*
    Draws a high resolution pressure field by bilinear interpolation of the off-grid pressures.

    @param grid
    @param res Target resolution of the field
*/
void Renderer::drawPressureFieldHR(const Grid& grid, sf::Vector2i res) {
    sf::Vector2i gridSize = {grid.getWidth(), grid.getHeight()};
    sf::Vector2f cellSize = {parameter.targetRect.size.x / float(gridSize.x), parameter.targetRect.size.y / float(gridSize.y)};
    sf::Vector2f stride = {gridSize.x / float(res.x), gridSize.y / float(res.y)};

    for(int i = 0; i < res.x; i++) {
        for(int j = 0; j < res.y; j++) {
            double p = grid.getPressure({stride.x * i, stride.y * j});
            auto position = parameter.targetRect.position + sf::Vector2f(stride.x * i * cellSize.x, stride.y * j * cellSize.y);
            sf::RectangleShape rect({stride.x * cellSize.x, stride.y * cellSize.y});
            rect.setPosition(position);
            rect.setFillColor(parameter.pressureColor.eval(p * 0.0005));

            window.draw(rect);
        }
    }
}