cmake_minimum_required(VERSION 4.2.3)
project(cfd)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SFML COMPONENTS Graphics REQUIRED)

include(FetchContent)
FetchContent_Declare(
  Kokkos
  URL https://github.com/kokkos/kokkos/archive/refs/tags/5.0.0.zip
)
FetchContent_MakeAvailable(Kokkos)

# set(Kokkos_ENABLE_HIP ON CACHE BOOL "" FORCE)
# set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)

# add_subdirectory(kokkos)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/graphics/renderer.cpp
    src/engine/engine.cpp
    src/engine/grid.cpp
    src/engine/solver.cpp
    src/engine/linearOperator.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/eigen
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    SFML::Graphics
    Kokkos::kokkos
)