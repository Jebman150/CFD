@group(0) @binding(0)
var<storage, read> input_x: array<f32>;

@group(0) @binding(1)
var<storage, read> input_y: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<storage, read> uparams: array<u32>;

const GRID_SIZE_X_IDX: u32 = 0;
const GRID_SIZE_Y_IDX: u32 = 1;

@compute @workgroup_size(64)
fn vector_magnitude_calculation(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let i = id.x;

    let nx = uparams[GRID_SIZE_X_IDX];
    let ny = uparams[GRID_SIZE_Y_IDX];

    if (i >= nx * ny) {
        return;
    }

    let x = i % nx;
    let y = i / nx;

    let left_x = y * (nx + 1u) + x;
    let right_x = left_x + 1u;

    let bottom_y = y * nx + x;
    let top_y = bottom_y + nx;

    let x_val =
        0.5 * (input_x[left_x] + input_x[right_x]);

    let y_val =
        0.5 * (input_y[bottom_y] + input_y[top_y]);

    output[i] = abs(x_val) + abs(y_val);
}