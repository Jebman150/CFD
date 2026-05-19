#pragma once

namespace engine {

namespace navigation {

    struct Index3D {
        int i, j, k;
    };

    enum Axis {
        X, Y, Z, Dim
    };

    enum Direction {
        Left,
        Right,
        Top,
        Bottom,
        Front,
        Back,
        NUM
    };

}

}