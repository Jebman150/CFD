@group(0) @binding(0)
var<storage, read_write> pressure: array<f32>;

struct Parameter {
    grid_size: vec2<u32>,
    dx: vec2<f32>,
    density: f32,
    dt: f32,
};

@group(0) @binding(2)
var<uniform> parameter: Parameter;

@group(0) @binding(3)
var<storage, read> is_solid_cell: array<u32>; // 0 => false, !0 => true

@group(0) @binding(5)
var<storage, read_write> ap: array<f32>;

@group(0) @binding(6)
var<storage, read_write> r: array<f32>;

@group(0) @binding(7)
var<storage, read> r_dot_r: array<f32>;

struct CGVars {
    alpha: f32,
    beta: f32,
    rs: f32,
    pad: f32
};

@group(0) @binding(8)
var<storage, read_write> cg_vars: CGVars;

@group(0) @binding(9)
var<storage, read_write> pap: array<f32>;

@group(0) @binding(10)
var<storage, read_write> p: array<f32>;

@group(0) @binding(11)
var<storage, read_write> reduction_input: array<f32>;

@compute @workgroup_size(64)
fn apply(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let i = id.x;
    let loc_i = local_id.x;

    let n = arrayLength(&p);

    let dx_2_inv = 1.0 / (parameter.dx[0] * parameter.dx[0]);
    let dy_2_inv = 1.0 / (parameter.dx[1] * parameter.dx[1]);

    if(i >= n) {
        return;
    }

    let nx = parameter.grid_size[0];
    let ny = parameter.grid_size[1];

    var result = 0.0;

    let x = i % nx;
    let y = i / nx;

    var x_neighbours = 0.0;
    if x > 0 && is_solid_cell[i - 1] == 0 {
        result -= dx_2_inv * p[i - 1];
        x_neighbours += 1.0;
    }
    if x < (nx - 1) && is_solid_cell[i + 1] == 0 {
        result -= dx_2_inv * p[i + 1];
        x_neighbours += 1.0;
    }

    var y_neighbours = 0.0;
    if y > 0 && is_solid_cell[i - nx] == 0 {
        result -= dy_2_inv * p[i - nx];
        y_neighbours += 1.0;
    }
    if y < (ny - 1) && is_solid_cell[i + ny] == 0 {
        result -= dy_2_inv * p[i + ny];
        y_neighbours += 1.0;
    }

    x_neighbours *= dx_2_inv;
    y_neighbours *= dy_2_inv;

    result += (x_neighbours + y_neighbours) * p[i];
    ap[i] = result;
    reduction_input[i+1] = p[i] * result;
    if i == 0 {
        reduction_input[0] = f32(n);
    }
}



@compute @workgroup_size(64)
fn update_p(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let rs_new = r_dot_r[0];
    var beta = 0.0;
    if(cg_vars.rs != 0) {
        beta = rs_new / cg_vars.rs;
    }

    if(i >= arrayLength(&p)) {
        return;
    }
    
    p[i] = r[i] + beta * p[i];

    if(i == 0) {
        cg_vars.rs = rs_new;
    }
}


@compute @workgroup_size(64)
fn update_xr(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    var alpha = 0.0;
    if(pap[0] != 0.0) {
        alpha = cg_vars.rs / pap[0];
    }
    let n = arrayLength(&pressure);

    if(i >= arrayLength(&pressure)) {
        return;
    }
    
    pressure[i] += alpha * p[i];
    r[i] -= alpha * ap[i];
    reduction_input[i+1] = r[i] * r[i];
    if(i == 0) {
        reduction_input[0] = f32(n);
    }
}