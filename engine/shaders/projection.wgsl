@group(0) @binding(0)
var<storage, read> pressure: array<f32>;

@group(0) @binding(1)
var<storage, read_write> velocity_x: array<f32>;

@group(0) @binding(2)
var<storage, read_write> velocity_y: array<f32>;

@group(0) @binding(3)
var<storage, read> uparams: array<u32>;

@group(0) @binding(4)
var<storage, read> fparams: array<f32>;

const GRID_SIZE_X_IDX: u32 = 0;
const GRID_SIZE_Y_IDX: u32 = 1;

const GRID_DX_IDX: u32 = 0;
const GRID_DY_IDX: u32 = 1;

const DELTA_T: u32 = 2;

const DENSITY: u32 = 3;

@compute @workgroup_size(64)
fn projection(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let i = id.x;

    let nx = uparams[GRID_SIZE_X_IDX];
    let ny = uparams[GRID_SIZE_Y_IDX];

    let x_face_count = (nx + 1u) * ny;
    let y_face_count = nx * (ny + 1u);
    let total_face_count = x_face_count + y_face_count;

    if (i >= total_face_count) {
        return;
    }

    let scale = fparams[DELTA_T] / fparams[DENSITY];

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
            / fparams[GRID_DX_IDX];

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
            / fparams[GRID_DY_IDX];

        velocity_y[j] += scale * pressure_gradient;
    }
}