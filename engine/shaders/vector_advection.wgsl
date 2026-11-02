@group(0) @binding(0)
var<storage, read> velocity_x: array<f32>;

@group(0) @binding(1)
var<storage, read> velocity_y: array<f32>;

@group(0) @binding(2)
var<storage, read> vector_field_in_x: array<f32>;

@group(0) @binding(3)
var<storage, read> vector_field_in_y: array<f32>;

@group(0) @binding(4)
var<storage, read_write> vector_field_out_x: array<f32>;

@group(0) @binding(5)
var<storage, read_write> vector_field_out_y: array<f32>;

@group(0) @binding(6)
var<storage> uparams: array<u32>;

@group(0) @binding(7)
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

fn get_mac_x_sampling_parameter(pos: vec2<f32>) -> vec2<f32> {
    let nx = uparams[GRID_SIZE_X_IDX];
    let ny = uparams[GRID_SIZE_Y_IDX];
    let dx = fparams[GRID_DX_IDX];
    let dy = fparams[GRID_DY_IDX];

    // Convert world position to Y-face grid coordinates.
    let grid_pos = vec2<f32>(
        pos.x / dx,
        pos.y / dy - 0.5
    );

    let gx = clamp(grid_pos.x, 0.0, f32(nx));
    let gy = clamp(grid_pos.y, 0.0, f32(ny - 1u));

    let x0 = u32(floor(gx));
    let y0 = u32(floor(gy));

    let tx = gx - f32(x0);
    let ty = gy - f32(y0);
    return vec2(tx, ty);
}

fn get_mac_y_sampling_parameter(pos: vec2<f32>) -> vec2<f32> {
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

    let tx = gx - f32(x0);
    let ty = gy - f32(y0);
    return vec2(tx, ty);
}

fn get_mac_x_sampling_idx(pos: vec2<f32>) -> vec4<u32> {
    let nx = uparams[GRID_SIZE_X_IDX];
    let ny = uparams[GRID_SIZE_Y_IDX];
    let dx = fparams[GRID_DX_IDX];
    let dy = fparams[GRID_DY_IDX];

    // Convert world position to Y-face grid coordinates.
    let grid_pos = vec2<f32>(
        pos.x / dx,
        pos.y / dy - 0.5
    );

    let gx = clamp(grid_pos.x, 0.0, f32(nx));
    let gy = clamp(grid_pos.y, 0.0, f32(ny - 1u));

    let x0 = u32(floor(gx));
    let y0 = u32(floor(gy));

    let x1 = min(x0 + 1u, nx);
    let y1 = min(y0 + 1u, ny - 1u);

    let tx = gx - f32(x0);
    let ty = gy - f32(y0);
    return vec4(x0, x1, y0, y1);
}

fn get_mac_y_sampling_idx(pos: vec2<f32>) -> vec4<u32> {
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
    return vec4(x0, x1, y0, y1);
}

fn sample_vector_field_x(pos: vec2<f32>) -> f32 {
    let idx = get_mac_x_sampling_idx(pos);
    let t = get_mac_x_sampling_parameter(pos);
    let nx = uparams[GRID_SIZE_X_IDX];

    let row_width = nx + 1u;

    let v00 = vector_field_in_x[idx[2] * row_width + idx[0]];
    let v10 = vector_field_in_x[idx[2] * row_width + idx[1]];
    let v01 = vector_field_in_x[idx[3] * row_width + idx[0]];
    let v11 = vector_field_in_x[idx[3] * row_width + idx[1]];

    return bilerp(v00, v10, v01, v11, t);
}

fn sample_vector_field_y(pos: vec2<f32>) -> f32 {
    let idx = get_mac_y_sampling_idx(pos);
    let t = get_mac_y_sampling_parameter(pos);
    let nx = uparams[GRID_SIZE_X_IDX];

    let v00 = vector_field_in_y[idx[2] * nx + idx[0]];
    let v10 = vector_field_in_y[idx[2] * nx + idx[1]];
    let v01 = vector_field_in_y[idx[3] * nx + idx[0]];
    let v11 = vector_field_in_y[idx[3] * nx + idx[1]];

    return bilerp(v00, v10, v01, v11, t);
}

fn sample_velocity_field_x(pos: vec2<f32>) -> f32 {
    let idx = get_mac_x_sampling_idx(pos);
    let t = get_mac_x_sampling_parameter(pos);
    let nx = uparams[GRID_SIZE_X_IDX];

    let row_width = nx + 1u;

    let v00 = velocity_x[idx[2] * row_width + idx[0]];
    let v10 = velocity_x[idx[2] * row_width + idx[1]];
    let v01 = velocity_x[idx[3] * row_width + idx[0]];
    let v11 = velocity_x[idx[3] * row_width + idx[1]];

    return bilerp(v00, v10, v01, v11, t);
}

fn sample_velocity_field_y(pos: vec2<f32>) -> f32 {
    let idx = get_mac_y_sampling_idx(pos);
    let t = get_mac_y_sampling_parameter(pos);
    let nx = uparams[GRID_SIZE_X_IDX];

    let v00 = velocity_y[idx[2] * nx + idx[0]];
    let v10 = velocity_y[idx[2] * nx + idx[1]];
    let v01 = velocity_y[idx[3] * nx + idx[0]];
    let v11 = velocity_y[idx[3] * nx + idx[1]];

    return bilerp(v00, v10, v01, v11, t);
}


// Sample the vector field at a world-space position.
fn sample_velocity(pos: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        sample_velocity_field_x(pos),
        sample_velocity_field_y(pos)
    );
}


@compute @workgroup_size(64)
fn vector_advection(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let i = id.x;

    let nx = uparams[GRID_SIZE_X_IDX];
    let ny = uparams[GRID_SIZE_Y_IDX];
    let dx = fparams[GRID_DX_IDX];
    let dy = fparams[GRID_DY_IDX];
    let dt = fparams[DELTA_T];

    let x_face_count = (nx + 1u) * ny;
    let y_face_count = nx * (ny + 1u);
    let total_face_count = x_face_count + y_face_count;

    if (i >= total_face_count) {
        return;
    }

    if (i < x_face_count) {
        let x = i % (nx + 1u);
        let y = i / (nx + 1u);

        // World position of this X face.
        let pos = vec2<f32>(
            f32(x) * dx,
            (f32(y) + 0.5) * dy
        );

        let vel = sample_velocity(pos);
        let prev_pos = pos - dt * vel;

        vector_field_out_x[i] = sample_vector_field_x(prev_pos);
    } else {
        let j = i - x_face_count;
        let x = j % nx;
        let y = j / nx;

        // World position of this Y face.
        let pos = vec2<f32>(
            (f32(x) + 0.5) * dx,
            f32(y) * dy
        );

        let vel = sample_velocity(pos);
        let prev_pos = pos - dt * vel;

        vector_field_out_y[j] = sample_vector_field_y(prev_pos);
    }
}