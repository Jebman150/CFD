@group(0) @binding(0)
var<storage, read> p: array<f32>;

@group(0) @binding(1)
var<storage, read> ap: array<f32>;

@group(0) @binding(2)
var<storage, read> alpha: array<f32>;

@group(0) @binding(3)
var<storage, read_write> x: array<f32>;

@group(0) @binding(4)
var<storage, read_write> r: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if(i >= arrayLength(&x)) {
        return;
    }
    
    x[i] += alpha[0] * p[i];
    r[i] -= alpha[0] * ap[i];
}