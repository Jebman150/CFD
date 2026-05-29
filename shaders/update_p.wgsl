@group(0) @binding(0)
var<storage, read> r: array<f32>;

@group(0) @binding(1)
var<storage, read> beta: array<f32>;

@group(0) @binding(2)
var<storage, read_write> p: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if(i >= arrayLength(&p)) {
        return;
    }
    
    p[i] = r[i] + beta[0] * p[i];
}