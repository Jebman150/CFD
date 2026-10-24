@group(0) @binding(0)
var<storage, read> pressure: array<f32>;

@group(0) @binding(1)
var<storage, read_write> velocity_x: array<f32>;

@group(0) @binding(2)
var<storage, read_write> velocity_y: array<f32>;

struct Parameter {
    grid_size: vec2<u32>,
    dx: vec2<f32>,
    density: f32,
    dt: f32
};

@group(0) @binding(3)
var<uniform> parameter: Parameter;

@group(0) @binding(4)
var<storage, read_write> abs_velocity: array<f32>;

@compute @workgroup_size(64)
fn velocity_projection(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let i = id.x;

    let nx = parameter.grid_size.x;
    let ny = parameter.grid_size.y;

    let x_face_count = (nx + 1u) * ny;
    let y_face_count = nx * (ny + 1u);
    let total_face_count = x_face_count + y_face_count;

    if (i >= total_face_count) {
        return;
    }

    let scale = parameter.dt / parameter.density;

    if (i < x_face_count) {
        let x = i % (nx + 1u);
        let y = i / (nx + 1u);

        // Preserve boundary values.
        if (x == 0u || x == nx) {
            return;
        }

        let left_cell = y * nx + (x - 1u);
        let right_cell = y * nx + x;

        let pressure_gradient =
            (pressure[right_cell] - pressure[left_cell])
            / parameter.dx.x;

        velocity_x[i] += scale * pressure_gradient;
    } else {
        let j = i - x_face_count;
        let x = j % nx;
        let y = j / nx;

        if (y == 0u || y == ny) {
            return;
        }

        let bottom_cell = (y - 1u) * nx + x;
        let top_cell = y * nx + x;

        let pressure_gradient =
            (pressure[top_cell] - pressure[bottom_cell])
            / parameter.dx.y;

        velocity_y[j] += scale * pressure_gradient;
    }
}

@compute @workgroup_size(64)
fn abs_velocity_value_calculation(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let i = id.x;

    let nx = parameter.grid_size.x;
    let ny = parameter.grid_size.y;

    if (i >= nx * ny) {
        return;
    }

    let x = i % nx;
    let y = i / nx;

    let left_x = y * (nx + 1u) + x;
    let right_x = left_x + 1u;

    let bottom_y = y * nx + x;
    let top_y = bottom_y + nx;

    let vel_x =
        0.5 * (velocity_x[left_x] + velocity_x[right_x]);

    let vel_y =
        0.5 * (velocity_y[bottom_y] + velocity_y[top_y]);

    abs_velocity[i] = abs(vel_x) + abs(vel_y);
}