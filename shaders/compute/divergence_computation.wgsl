struct Parameter {
    grid_size: vec2<u32>,
    dx: vec2<f32>,
    density: f32,
    dt: f32,
};

@group(0) @binding(0)
var<storage, read> velocity_x: array<f32>;

@group(0) @binding(1)
var<storage, read> velocity_y: array<f32>;

@group(0) @binding(2)
var<storage, read_write> divergence: array<f32>;

@group(0) @binding(3)
var<uniform> params: Parameter;

@compute @workgroup_size(64)
fn compute_divergence(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let index = global_id.x;
    let nx = params.grid_size.x;
    let ny = params.grid_size.y;
    let cell_count = nx * ny;

    if (index >= cell_count) {
        return;
    }

    let cell_x = index % nx;
    let cell_y = index / nx;

    var sum = 0.0;

    // X-axis contribution: (right - left) / dx.x
    let left_x = velocity_x[cell_y * (nx + 1) + cell_x];
    let right_x = velocity_x[cell_y * (nx + 1) + cell_x + 1u];

    sum += (right_x - left_x) / params.dx.x;

    // Y-axis contribution: (bottom - top) / dx.y
    let top_y = velocity_y[cell_y * nx + cell_x];
    let bottom_y = velocity_y[(cell_y + 1) * nx + cell_x];

    sum += (bottom_y - top_y) / params.dx.y;

    divergence[index] = sum;
}