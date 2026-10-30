@group(0) @binding(0)
var<storage, read_write> divergence: array<f32>;

struct Parameter {
    grid_size: vec2<u32>,
    dx: vec2<f32>,
    density: f32,
    dt: f32,
};

@group(0) @binding(1)
var<uniform> parameter: Parameter;

@group(0) @binding(2)
var<storage, read_write> pressure: array<f32>;

@group(0) @binding(3)
var<storage, read> r_dot_r: array<f32>;

@group(0) @binding(4)
var<storage, read_write> r: array<f32>;

struct CGVars {
    alpha: f32,
    beta: f32,
    rs: f32,
    pad: f32,
}

@group(0) @binding(5)
var<storage, read_write> cg_vars: CGVars;

@group(0) @binding(6)
var<storage, read_write> reduction_input: array<f32>;

@group(0) @binding(7)
var<storage, read_write> p: array<f32>;

@compute @workgroup_size(64)
fn process_divergence(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let i = id.x;
    let n = arrayLength(&divergence);

    if i >= n {
        return;
    }

    let d = divergence[i];
    let scaled = d * parameter.density / parameter.dt;

    divergence[i] = scaled;
    pressure[i] = 0.0;
    p[i] = scaled;
    r[i] = scaled;
    reduction_input[i+1] = scaled * scaled;
    if i == 0{
        reduction_input[0] = f32(n);
    }
}

@compute @workgroup_size(64)
fn set_initial_cg_vars(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let i = id.x;

    if i != 0 {
        return;
    }
    cg_vars.rs = r_dot_r[0];
    cg_vars.alpha = 0.0;
    cg_vars.beta = 0.0;
}