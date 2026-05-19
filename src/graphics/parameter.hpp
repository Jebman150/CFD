#pragma once

#include <SFML/Graphics.hpp>
#include <cmath>

struct ColorGradient {
    sf::Color colorLow;
    sf::Color colorHigh;
    bool negative;

    sf::Color eval(float t) const {
        if(negative) t = (t + 1) / 2;
        t = std::clamp(t, 0.f, 1.f);
        return sf::Color(
            colorLow.r + t *(colorHigh.r-colorLow.r),
            colorLow.g + t *(colorHigh.g-colorLow.g),
            colorLow.b + t *(colorHigh.b-colorLow.b),
            colorLow.a + t *(colorHigh.a-colorLow.a)
        );
    }
};

struct GraphicalParameter {
    const sf::Color xAxis = sf::Color::Red;
    const sf::Color yAxis = sf::Color::Blue;
    const sf::Color zAxis = sf::Color::Green;

    const sf::Color gridCellColor = sf::Color(90, 90, 90, 20);
    const sf::Color backgroundColor = sf::Color(10, 10, 20);
    const ColorGradient pressureColor = {
        .colorLow = sf::Color::Blue,
        .colorHigh = sf::Color::Red,
        .negative = true
    };
    const ColorGradient velocityColor = {
        .colorLow = sf::Color::Green,
        .colorHigh = sf::Color::Red,
        .negative = false
    };
    const ColorGradient divergenceColor = {
        .colorLow = sf::Color::Red,
        .colorHigh = sf::Color::Blue,
        .negative = true
    };
    const ColorGradient smokeColor = {
        .colorLow = sf::Color::Black,
        .colorHigh = sf::Color::White,
        .negative = false
    };

    const sf::Vector2u windowSize = {1000, 1000};
    const sf::FloatRect targetRect = {{100, 100}, {800, 800}};

};