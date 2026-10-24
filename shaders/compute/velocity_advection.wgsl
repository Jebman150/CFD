@group(0) @binding(0)
var<storage, read> velocity_x_in: array<f32>;

@group(0) @binding(1)
var<storage, read> velocity_y_in: array<f32>;

@group(0) @binding(2)
var<storage, read_write> velocity_x_out: array<f32>;

@group(0) @binding(3)
var<storage, read_write> velocity_y_out: array<f32>;

struct Parameter {
    grid_size: vec2<u32>,
    delta_x: vec2<f32>,
    density: f32,
    dt: f32,
};

@group(0) @binding(4)
var<uniform> parameter: Parameter;

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

fn sample_velocity_x(pos: vec2<f32>) -> f32 {
    let nx = parameter.grid_size.x;
    let ny = parameter.grid_size.y;
    let dx = parameter.delta_x;

    // Convert world position to X-face grid coordinates.
    let grid_pos = vec2<f32>(
        pos.x / dx.x,
        pos.y / dx.y - 0.5
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

    let v00 = velocity_x_in[y0 * row_width + x0];
    let v10 = velocity_x_in[y0 * row_width + x1];
    let v01 = velocity_x_in[y1 * row_width + x0];
    let v11 = velocity_x_in[y1 * row_width + x1];

    return bilerp(v00, v10, v01, v11, vec2<f32>(tx, ty));
}


// Sample Y velocity on the staggered Y-face grid.
// Grid dimensions: nx x (ny + 1).
// Face positions: ((x + 0.5) * dx, y * dy).
fn sample_velocity_y(pos: vec2<f32>) -> f32 {
    let nx = parameter.grid_size.x;
    let ny = parameter.grid_size.y;
    let dx = parameter.delta_x;

    // Convert world position to Y-face grid coordinates.
    let grid_pos = vec2<f32>(
        pos.x / dx.x - 0.5,
        pos.y / dx.y
    );

    let gx = clamp(grid_pos.x, 0.0, f32(nx - 1u));
    let gy = clamp(grid_pos.y, 0.0, f32(ny));

    let x0 = u32(floor(gx));
    let y0 = u32(floor(gy));

    let x1 = min(x0 + 1u, nx - 1u);
    let y1 = min(y0 + 1u, ny);

    let tx = gx - f32(x0);
    let ty = gy - f32(y0);

    let v00 = velocity_y_in[y0 * nx + x0];
    let v10 = velocity_y_in[y0 * nx + x1];
    let v01 = velocity_y_in[y1 * nx + x0];
    let v11 = velocity_y_in[y1 * nx + x1];

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
fn velocity_advection(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let i = id.x;

    let nx = parameter.grid_size.x;
    let ny = parameter.grid_size.y;
    let dx = parameter.delta_x;
    let dt = parameter.dt;

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
            f32(x) * dx.x,
            (f32(y) + 0.5) * dx.y
        );

        let vel = sample_velocity(pos);
        let prev_pos = pos - dt * vel;

        velocity_x_out[i] = sample_velocity_x(prev_pos);
    } else {
        let j = i - x_face_count;
        let x = j % nx;
        let y = j / nx;

        // World position of this Y face.
        let pos = vec2<f32>(
            (f32(x) + 0.5) * dx.x,
            f32(y) * dx.y
        );

        let vel = sample_velocity(pos);
        let prev_pos = pos - dt * vel;

        velocity_y_out[j] = sample_velocity_y(prev_pos);
    }
}