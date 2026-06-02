@group(0) @binding(0)
var<storage, read> p: array<f32>;

@group(0) @binding(1)
var<storage, read> ap: array<f32>;

@group(0) @binding(2)
var<storage, read> pap: array<f32>;

struct Parameter {
    alpha: f32,
    beta: f32,
    rs: f32,
    pad: f32
};

@group(0) @binding(3)
var<storage, read> param: Parameter;

@group(0) @binding(4)
var<storage, read_write> x: array<f32>;

@group(0) @binding(5)
var<storage, read_write> r: array<f32>;

@group(0) @binding(6)
var<storage, read_write> partial1: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    var alpha = 0.0;
    if(pap[0] != 0.0) {
        alpha = param.rs / pap[0];
    }
    let n = arrayLength(&x);

    if(i >= arrayLength(&x)) {
        return;
    }
    
    x[i] += alpha * p[i];
    r[i] -= alpha * ap[i];
    partial1[i+1] = r[i] * r[i];
    if(i == 0) {
        partial1[0] = f32(n);
    }
}