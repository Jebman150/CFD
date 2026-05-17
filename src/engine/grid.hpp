#pragma once

#include <Eigen/Dense>
#include <vector>

struct GridCoord {
    double x, y;
};

struct Index2D {
    int i, j;
};

class Grid {
    Eigen::MatrixXd u;
    Eigen::MatrixXd v;
    Eigen::MatrixXd p;

    Eigen::MatrixXd uBuffer;
    Eigen::MatrixXd vBuffer;

    double deltaX;
    double deltaY;

    int width, height;

    double getExactP(int x, int y) const;
    double getExactU(int x, int y) const;
    double getExactV(int x, int y) const;
public:
    void initialize(Eigen::Vector2i _gridSize);

    void updateU(Index2D idx, double val);
    void updateV(Index2D idx, double val);

    void applyBuffer();
    void flushBuffer();

    // Setter
    void setPressure(Index2D idx, double val);
    void setVelocityU(Index2D idx, double val);
    void setVelocityV(Index2D idx, double val);

    void bufferVelocityU(Index2D idx, double val);
    void bufferVelocityV(Index2D idx, double val);

    void setP(Eigen::MatrixXd mat) { p = mat; }
    void setU(Eigen::MatrixXd mat) { u = mat; }
    void setV(Eigen::MatrixXd mat) { v = mat; }

    // Computational getter
    Eigen::Vector2d interpolateVelocity(GridCoord coord) const;
    double getDivergence(GridCoord coord) const;

    // Getter
    const Eigen::MatrixXd& getPressureValues() const { return p; }
    const Eigen::MatrixXd& getVelocityXValues() const { return u; }
    const Eigen::MatrixXd& getVelocityYValues() const { return v; }

    double getPressure(GridCoord coord) const;
    double getVelocityU(GridCoord coord) const;
    double getVelocityV(GridCoord coord) const;

    int getWidth() const { return width; }
    int getHeight() const { return height; }

    double getDx() const { return deltaX; }
    double getDy() const { return deltaY; }

    GridCoord coordFromCellIndex(Index2D idx) { return {idx.i + 0.5, idx.j + 0.5}; }
    GridCoord coordFromUIndex(Index2D idx) { return {double(idx.i), idx.j + 0.5}; }
    GridCoord coordFromVIndex(Index2D idx) { return {idx.i + 0.5, double(idx.j)}; }
};