#include "renderer.hpp"
#include "engine/grid.hpp"
#include "engine/navigation.hpp"
#include<iostream>

namespace renderer {

using engine::Grid;
using engine::ConstFaceView;
using engine::ConstCellView;
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


    gridSize = {grid.getSize()[0], grid.getSize()[1]};
    cellSize = {parameter.targetRect.size.x / float(gridSize.x), parameter.targetRect.size.y / float(gridSize.y)};

    drawCoordinateSystem();
    drawCells(grid);
    drawFaces(grid);

    switch(visMode) {
        case VisMode::Pressure:
            drawScalarHRField([&grid] (float x, float y) {
                GridCoord coord;
                coord.coord[0] = x;
                coord.coord[1] = y;
                return grid.getScalarField(ScalarFieldID::Pressure, coord);
            }, resolution);
            break;
        case VisMode::Speed:
            drawScalarHRField([&grid] (float x, float y) {
                GridCoord coord;
                coord.coord[0] = x;
                coord.coord[1] = y;
                return grid.interpolateVelocity(coord).norm();
            }, resolution);
            break;
        case VisMode::Divergence:
            drawDivergenceField(grid);
            break;
        case VisMode::Smoke:
            drawScalarHRField([&grid] (float x, float y) {
                GridCoord coord;
                coord.coord[0] = x;
                coord.coord[1] = y;
                if(grid.getCellType(MultiIndex({int(x), int(y)}, grid.getCellContext())) == engine::CellType::Solid) {
                    return 0.f;
                }
                return grid.getScalarField(ScalarFieldID::Smoke, coord);
            }, resolution);
            break;
        default:
            break;
    }

    switch(vecMode) {
        case VectorFieldMode::VelocityHR:
            drawVelocityFieldHR(grid, resolution);
            break;
        case VectorFieldMode::VelocityDebug:
            drawVelocityPlaneDebug(grid);
        default:
            break;
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
    grid.queryCells([this](ConstCellView cell) {
            auto rect = sf::RectangleShape(cellSize);
            rect.setFillColor((cell.type == engine::CellType::Solid) ? parameter.solidCellColor : parameter.fluidCellColor);
            auto indices = cell.idx.getIndices();
            rect.setPosition(parameter.targetRect.position + sf::Vector2f{indices[0]*cellSize.x, indices[1]*cellSize.y});
            
            window.draw(rect);
    });
}

/*
    Draws the outline of the discretization cells.

    @param grid
*/
void Renderer::drawFaces(const Grid& grid) {
    grid.queryFaces(
    [this](ConstFaceView face) {
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
        auto indices = face.idx.getIndices();
        rect.setPosition(parameter.targetRect.position + sf::Vector2f{indices[0]*cellSize.x, indices[1]*cellSize.y});

        window.draw(rect);
    });
}

/*
    Draws the divergence of the given grid.

    OUT OF ORDER

    @param grid
*/
void Renderer::drawDivergenceField(const Grid& grid) {
    /*for(float x = 0.5; x < gridSize.x; x++) {
        for(float y = 0.5; y < gridSize.y; y++) {
            auto rect = sf::RectangleShape(cellSize);
            float value = grid.getDivergence({x, y, 0});
            rect.setFillColor(parameter.pressureColor.eval(value));
            rect.setPosition(parameter.targetRect.position + sf::Vector2f{(x-0.5)*cellSize.x, (y-0.5)*cellSize.y});

            window.draw(rect);
        }
    }*/
}

/*
    Draws the velocities of the z plane in the grid
*/
void Renderer::drawVelocityPlaneDebug(const Grid& grid) {
    grid.queryFaces([this](ConstFaceView face) {
        sf::Vector2f size(10, 10);
        sf::Vector2f offset(-5, -5);
        sf::Vector2f origin(2.5f, 0);
        switch(face.axis) {
            case Axis::X: 
                size = {face.value * 10, 5.f};
                offset = {0, cellSize.y * 0.5f};
                origin = {0, 2.5f};
                break;
            case Axis::Y: 
                size = {5.f, face.value * 10}; 
                offset = {cellSize.x * 0.5f, 0};
                origin = {2.5f, 0};
                break;
            default:
                std::cout << "WARNING: Invalid face in tangent faces query" << std::endl;
        }
        auto rect = sf::RectangleShape(size);
        rect.setFillColor(parameter.velocityColor.eval(std::fabs(face.value)));
        rect.setOrigin(origin);
        auto indices = face.idx.getIndices();
        rect.setPosition(parameter.targetRect.position + offset + sf::Vector2f{indices[0]*cellSize.x, indices[1]*cellSize.y});

        window.draw(rect);
    });
}

/*
    Draws a high resolution velocity field by bilinear interpolation of the off-grid velocities.

    @param grid
    @param res Target resolution of the field
*/
void Renderer::drawVelocityFieldHR(const Grid& grid, sf::Vector2i res) {
    sf::Vector2f stride = {gridSize.x / float(res.x), gridSize.y / float(res.y)};

    for(int i = 0; i < res.x; i++) {
        for(int j = 0; j < res.y; j++) {
            GridCoord coord;
            coord.coord[0] = stride.x * i;
            coord.coord[1] = stride.y * j;
            auto vel = grid.interpolateVelocity(coord);
            auto position = parameter.targetRect.position + sf::Vector2f(stride.x * i * cellSize.x, stride.y * j * cellSize.y);
            sf::Vertex line[] = {
                {position, sf::Color::Cyan},
                {position + sf::Vector2f(vel.x(), vel.y()) * 5.f, sf::Color::Cyan}
            };

            window.draw(line, 2, sf::PrimitiveType::Lines);
        }
    }
}

}