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
    let cell_count = params.grid_size.x * params.grid_size.y;

    if (index >= cell_count) {
        return;
    }

    let x = index % params.grid_size.x;
    let y = index / params.grid_size.x;

    var sum = 0.0;

    // X-axis contribution: (right - left) / dx.x
    let left_x = velocity_x[index];
    let right_x = velocity_x[index + 1u];

    sum += (right_x - left_x) / params.dx.x;

    // Y-axis contribution: (top - bottom) / dx.y
    let bottom_y = velocity_y[index];
    let top_y = velocity_y[index + params.grid_size.x];

    sum += (top_y - bottom_y) / params.dx.y;

    divergence[index] = sum;
}