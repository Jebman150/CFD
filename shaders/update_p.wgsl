@group(0) @binding(0)
var<storage, read> r: array<f32>;

@group(0) @binding(1)
var<storage, read> r_dot_r: array<f32>;

struct Parameter {
    alpha: f32,
    beta: f32,
    rs: f32,
    pad: f32
};

@group(0) @binding(2)
var<storage, read_write> param: Parameter;

@group(0) @binding(3)
var<storage, read_write> p: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let rs_new = r_dot_r[0];
    var beta = 0.0;
    if(param.rs != 0) {
        beta = rs_new / param.rs;
    }

    if(i >= arrayLength(&p)) {
        return;
    }
    
    p[i] = r[i] + beta * p[i];

    if(i == 0) {
        param.rs = rs_new;
    }
}