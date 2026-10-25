@group(0) @binding(0)
var<storage, read_write> pressure: array<f32>;

@group(0) @binding(1)
var<storage, read> x: array<f32>;

@compute @workgroup_size(64)
fn read_back_solution(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let i = id.x;
    let n = arrayLength(&pressure);

    if i >= n {
        return;
    }

    pressure[i] = x[i];
}