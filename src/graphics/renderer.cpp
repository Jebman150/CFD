#include "renderer.hpp"
#include<iostream>

namespace renderer {

using engine::Grid;
using engine::ConstFaceView;
using engine::ConstCellView;
using engine::Plane;
using engine::ScalarFieldID;
using namespace engine::navigation;

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

    drawCoordinateSystem();
    drawCells(grid);
    drawFaces(grid);

    switch(visMode) {
        case VisMode::Pressure:
            drawPressureFieldHR(grid, resolution);
            break;
        case VisMode::Speed:
            drawSpeedFieldHR(grid, resolution);
            break;
        case VisMode::Divergence:
            drawDivergenceField(grid);
            break;
        case VisMode::Smoke:
            drawSmokeFieldHR(grid, resolution);
            break;
    }

    switch(vecMode) {
        case VectorFieldMode::VelocityHR:
            drawVelocityFieldHR(grid, resolution);
            break;
        case VectorFieldMode::VelocityDebug:
            drawVelocityPlaneDebug(grid);
    }

    window.display();
}

void Renderer::drawCoordinateSystem() {
    sf::Vertex axis[] = {
        {parameter.targetRect.position / 2.f, parameter.xAxis},
        {parameter.targetRect.position / 2.f + sf::Vector2f{20, 0}, parameter.xAxis},
        {parameter.targetRect.position / 2.f, parameter.yAxis},
        {parameter.targetRect.position / 2.f + sf::Vector2f{0, 20}, parameter.yAxis},
        {parameter.targetRect.position / 2.f, parameter.zAxis},
        {parameter.targetRect.position / 2.f + sf::Vector2f{10, 10}, parameter.zAxis}
    };
    window.draw(axis, 6, sf::PrimitiveType::Lines);
}

/*
    Draws the background of the discretization cells.

    @param grid
*/
void Renderer::drawCells(const Grid& grid) {
    sf::Vector2i gridSize = {grid.getWidth(), grid.getHeight()};
    sf::Vector2f cellSize = {parameter.targetRect.size.x / float(gridSize.x), parameter.targetRect.size.y / float(gridSize.y)};

    grid.queryCells([&cellSize, this](ConstCellView cell) {
        auto rect = sf::RectangleShape(cellSize);
        rect.setFillColor((cell.type == engine::CellType::Solid) ? parameter.solidCellColor : parameter.fluidCellColor);
        rect.setPosition(parameter.targetRect.position + sf::Vector2f{cell.idx.i*cellSize.x, cell.idx.j*cellSize.y});

        window.draw(rect);
    }, [](ConstCellView cell) {
        return cell.idx.k == 0;
    });
}

/*
    Draws the outline of the discretization cells.

    @param grid
*/
void Renderer::drawFaces(const Grid& grid) {
    sf::Vector2i gridSize = {grid.getWidth(), grid.getHeight()};
    sf::Vector2f cellSize = {parameter.targetRect.size.x / float(gridSize.x), parameter.targetRect.size.y / float(gridSize.y)};

    grid.queryTangentFaces(
        {Axis::Z, 0},
    [&cellSize, this](ConstFaceView face) {
        if(face.type == engine::FaceType::Solid_Solid) return;
        sf::Vector2f size;
        sf::Vector2f offset;
        if(face.axis == Axis::X) {
            size = {4, cellSize.y};
            offset = {2, 0};
        } else {
            size = {cellSize.x, 4};
            offset = {0, 2};
        }
        auto rect = sf::RectangleShape(size);
        rect.setOrigin(offset);
        rect.setFillColor((face.type == engine::FaceType::Fluid_Fluid) ? parameter.fluidFaceColor : parameter.solidFaceColor);
        rect.setPosition(parameter.targetRect.position + sf::Vector2f{face.idx.i*cellSize.x, face.idx.j*cellSize.y});

        window.draw(rect);
    });
}

/*
    Draws the divergence of the given grid.

    @param grid
*/
void Renderer::drawDivergenceField(const Grid& grid) {
    sf::Vector2i gridSize = {grid.getWidth(), grid.getHeight()};
    sf::Vector2f cellSize = {parameter.targetRect.size.x / float(gridSize.x), parameter.targetRect.size.y / float(gridSize.y)};

    for(float x = 0.5; x < gridSize.x; x++) {
        for(float y = 0.5; y < gridSize.y; y++) {
            auto rect = sf::RectangleShape(cellSize);
            float value = grid.getDivergence({x, y, 0});
            rect.setFillColor(parameter.pressureColor.eval(value));
            rect.setPosition(parameter.targetRect.position + sf::Vector2f{(x-0.5)*cellSize.x, (y-0.5)*cellSize.y});

            window.draw(rect);
        }
    }
}

/*
    Draws the velocities of the z plane in the grid
*/
void Renderer::drawVelocityPlaneDebug(const Grid& grid) {
    sf::Vector2i gridSize = {grid.getDimensions(Axis::Z).x(), grid.getDimensions(Axis::Z).y()};
    sf::Vector2f cellSize = {parameter.targetRect.size.x / float(gridSize.x), parameter.targetRect.size.y / float(gridSize.y)};

    Plane plane = {Axis::Z, 0};
    grid.queryTangentFaces(plane, [this, cellSize](ConstFaceView face) {
        sf::Vector2f size(10, 10);
        sf::Vector2f offset(-5, -5);
        switch(face.axis) {
            case Axis::X: 
                size = {face.value * 10, 5.f};
                offset = {0, cellSize.y * 0.5f};
                break;
            case Axis::Y: 
                size = {5.f, face.value * 10}; 
                offset = {cellSize.x * 0.5f, 0};
                break;
        }
        auto rect = sf::RectangleShape(size);
        rect.setFillColor(parameter.velocityColor.eval(face.value));
        rect.setPosition(parameter.targetRect.position + offset + sf::Vector2f{face.idx.i*cellSize.x, face.idx.j*cellSize.y});

        window.draw(rect);
    });
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
            auto vel = grid.interpolateVelocity({stride.x * i, stride.y * j, 0});
            auto position = parameter.targetRect.position + sf::Vector2f(stride.x * i * cellSize.x, stride.y * j * cellSize.y);
            sf::Vertex line[] = {
                {position, sf::Color::Cyan},
                {position + sf::Vector2f(vel.x(), vel.y()) * 2.f, sf::Color::Cyan}
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
            double p = grid.getScalarField(ScalarFieldID::Pressure, {stride.x * i, stride.y * j, 0});
            auto position = parameter.targetRect.position + sf::Vector2f(stride.x * i * cellSize.x, stride.y * j * cellSize.y);
            sf::RectangleShape rect({stride.x * cellSize.x, stride.y * cellSize.y});
            rect.setPosition(position);
            rect.setFillColor(parameter.pressureColor.eval(p * 0.0005));

            window.draw(rect);
        }
    }
}

/*
    Draws a high resolution speed field by bilinear interpolation of the off-grid speeds.

    @param grid
    @param res Target resolution of the field
*/
void Renderer::drawSpeedFieldHR(const Grid& grid, sf::Vector2i res) {
    sf::Vector2i gridSize = {grid.getWidth(), grid.getHeight()};
    sf::Vector2f cellSize = {parameter.targetRect.size.x / float(gridSize.x), parameter.targetRect.size.y / float(gridSize.y)};
    sf::Vector2f stride = {gridSize.x / float(res.x), gridSize.y / float(res.y)};

    for(int i = 0; i < res.x; i++) {
        for(int j = 0; j < res.y; j++) {
            double vel = grid.interpolateVelocity({stride.x * i, stride.y * j, 0}).norm();
            auto position = parameter.targetRect.position + sf::Vector2f(stride.x * i * cellSize.x, stride.y * j * cellSize.y);
            sf::RectangleShape rect({stride.x * cellSize.x, stride.y * cellSize.y});
            rect.setPosition(position);
            rect.setFillColor(parameter.velocityColor.eval(vel * 0.1));

            window.draw(rect);
        }
    }
}

/*
    Draws a high resolution speed field by bilinear interpolation of the off-grid speeds.

    @param grid
    @param res Target resolution of the field
*/
void Renderer::drawSmokeFieldHR(const Grid& grid, sf::Vector2i res) {
    sf::Vector2i gridSize = {grid.getWidth(), grid.getHeight()};
    sf::Vector2f cellSize = {parameter.targetRect.size.x / float(gridSize.x), parameter.targetRect.size.y / float(gridSize.y)};
    sf::Vector2f stride = {gridSize.x / float(res.x), gridSize.y / float(res.y)};

    for(int i = 0; i < res.x; i++) {
        for(int j = 0; j < res.y; j++) {
            float val = grid.getScalarField(ScalarFieldID::Smoke, {stride.x * i, stride.y * j, 0});
            auto position = parameter.targetRect.position + sf::Vector2f(stride.x * i * cellSize.x, stride.y * j * cellSize.y);
            sf::RectangleShape rect({stride.x * cellSize.x, stride.y * cellSize.y});
            rect.setPosition(position);
            rect.setFillColor(parameter.smokeColor.eval(val));

            window.draw(rect);
        }
    }
}

}