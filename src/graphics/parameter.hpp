#pragma once

#include <SFML/Graphics.hpp>

struct ColorGradient {
    sf::Color colorLow;
    sf::Color colorHigh;

    sf::Color eval(float t) const {
        return sf::Color(
            colorLow.r + t *(colorHigh.r-colorLow.r),
            colorLow.g + t *(colorHigh.g-colorLow.g),
            colorLow.b + t *(colorHigh.b-colorLow.b),
            colorLow.a + t *(colorHigh.a-colorLow.a)
        );
    }
};

struct GraphicalParameter {
    const sf::Color gridCellColor = sf::Color(90, 90, 90);
    const sf::Color backgroundColor = sf::Color(10, 10, 20);
    const ColorGradient pressureColor = {
        colorLow: sf::Color::Blue,
        colorHigh: sf::Color::Red
    };
    const ColorGradient velocityColor = {
        colorLow: sf::Color::Green,
        colorHigh: sf::Color::Red
    };

    const sf::Vector2f windowSize = {1000, 1000};
    const sf::FloatRect targetRect = {{100, 100}, {800, 800}};

};