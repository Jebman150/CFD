cmake_minimum_required(VERSION 4.2.3)
project(cfd)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SFML COMPONENTS Graphics REQUIRED)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/graphics/renderer.cpp
    src/engine/engine.cpp
    src/engine/grid.cpp
    src/engine/pressureSolver.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/eigen
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    SFML::Graphics
)