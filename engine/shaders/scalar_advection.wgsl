@group(0) @binding(0)
var<storage, read> velocity_x: array<f32>;

@group(0) @binding(1)
var<storage, read> velocity_y: array<f32>;

@group(0) @binding(2)
var<storage, read> scalar_field_in: array<f32>;

@group(0) @binding(3)
var<storage, read_write> scalar_field_out: array<f32>;

@group(0) @binding(4)
var<storage> uparams: array<u32>;

@group(0) @binding(5)
var<storage> fparams: array<f32>;

const GRID_SIZE_X_IDX: u32 = 0;
const GRID_SIZE_Y_IDX: u32 = 1;

const GRID_DX_IDX: u32 = 0;
const GRID_DY_IDX: u32 = 1;

const DELTA_T: u32 = 2;

fn bilerp(
    v00: f32,
    v10: f32,
    v01: f32,
    v11: f32,
    t: vec2<f32>,
) -> f32 {
    let bottom = mix(v00, v10, t.x);
    let top = mix(v01, v11, t.x);
    return mix(bottom, top, t.y);
}

fn sample_scalar(pos: vec2<f32>) -> f32 {
    let nx = uparams[GRID_SIZE_X_IDX];
    let ny = uparams[GRID_SIZE_Y_IDX];
    let dx = fparams[GRID_DX_IDX];
    let dy = fparams[GRID_DY_IDX];

    // Convert world position to cell grid coordinates.
    let grid_pos = vec2<f32>(
        pos.x / dx - 0.5,
        pos.y / dy - 0.5
    );

    // Clamp to the available face-grid domain.
    let gx = clamp(grid_pos.x, 0.0, f32(nx - 1u));
    let gy = clamp(grid_pos.y, 0.0, f32(ny - 1u));

    let x0 = u32(floor(gx));
    let y0 = u32(floor(gy));

    let x1 = min(x0 + 1u, nx - 1u);
    let y1 = min(y0 + 1u, ny - 1u);

    let tx = gx - f32(x0);
    let ty = gy - f32(y0);

    let v00 = scalar_field_in[y0 * nx + x0];
    let v10 = scalar_field_in[y0 * nx + x1];
    let v01 = scalar_field_in[y1 * nx + x0];
    let v11 = scalar_field_in[y1 * nx + x1];

    return bilerp(v00, v10, v01, v11, vec2<f32>(tx, ty));
}

fn sample_velocity_x(pos: vec2<f32>) -> f32 {
    let nx = uparams[GRID_SIZE_X_IDX];
    let ny = uparams[GRID_SIZE_Y_IDX];
    let dx = fparams[GRID_DX_IDX];
    let dy = fparams[GRID_DY_IDX];

    // Convert world position to X-face grid coordinates.
    let grid_pos = vec2<f32>(
        pos.x / dx,
        pos.y / dy - 0.5
    );

    // Clamp to the available face-grid domain.
    let gx = clamp(grid_pos.x, 0.0, f32(nx));
    let gy = clamp(grid_pos.y, 0.0, f32(ny - 1u));

    let x0 = u32(floor(gx));
    let y0 = u32(floor(gy));

    let x1 = min(x0 + 1u, nx);
    let y1 = min(y0 + 1u, ny - 1u);

    let tx = gx - f32(x0);
    let ty = gy - f32(y0);

    let row_width = nx + 1u;

    let v00 = velocity_x[y0 * row_width + x0];
    let v10 = velocity_x[y0 * row_width + x1];
    let v01 = velocity_x[y1 * row_width + x0];
    let v11 = velocity_x[y1 * row_width + x1];

    return bilerp(v00, v10, v01, v11, vec2<f32>(tx, ty));
}


// Sample Y velocity on the staggered Y-face grid.
// Grid dimensions: nx x (ny + 1).
// Face positions: ((x + 0.5) * dx, y * dy).
fn sample_velocity_y(pos: vec2<f32>) -> f32 {
    let nx = uparams[GRID_SIZE_X_IDX];
    let ny = uparams[GRID_SIZE_Y_IDX];
    let dx = fparams[GRID_DX_IDX];
    let dy = fparams[GRID_DY_IDX];

    // Convert world position to Y-face grid coordinates.
    let grid_pos = vec2<f32>(
        pos.x / dx - 0.5,
        pos.y / dy
    );

    let gx = clamp(grid_pos.x, 0.0, f32(nx - 1u));
    let gy = clamp(grid_pos.y, 0.0, f32(ny));

    let x0 = u32(floor(gx));
    let y0 = u32(floor(gy));

    let x1 = min(x0 + 1u, nx - 1u);
    let y1 = min(y0 + 1u, ny);

    let tx = gx - f32(x0);
    let ty = gy - f32(y0);

    let v00 = velocity_y[y0 * nx + x0];
    let v10 = velocity_y[y0 * nx + x1];
    let v01 = velocity_y[y1 * nx + x0];
    let v11 = velocity_y[y1 * nx + x1];

    return bilerp(v00, v10, v01, v11, vec2<f32>(tx, ty));
}


// Sample the vector field at a world-space position.
fn sample_velocity(pos: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        sample_velocity_x(pos),
        sample_velocity_y(pos)
    );
}

@compute @workgroup_size(64)
fn scalar_advection(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let i = id.x;

    let nx = uparams[GRID_SIZE_X_IDX];
    let ny = uparams[GRID_SIZE_Y_IDX];
    let dx = fparams[GRID_DX_IDX];
    let dy = fparams[GRID_DY_IDX];
    let dt = fparams[DELTA_T];

    if i >= nx*ny {
        return;
    }

    let x = i % nx;
    let y = i / nx;

    let pos = vec2<f32>(
        (f32(x) + 0.5) * dx,
        (f32(y) + 0.5) * dy
    );

    let vel = sample_velocity(pos);
    let prev_pos = pos - dt * vel;

    scalar_field_out[i] = sample_scalar(prev_pos);
}