@group(0) @binding(0)
var<storage, read> dot_output: array<f32>;

struct Parameter {
    alpha: f32,
    beta: f32,
    rs: f32,
    pad: f32
};

@group(0) @binding(1)
var<storage, read_write> param: Parameter;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if(i != 0) {
        return;
    }
    
    let denom = dot_output[0];

    if (abs(denom) > 1e-20) {
        param.alpha = param.rs / denom;
    }
}