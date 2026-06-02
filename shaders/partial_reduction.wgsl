@group(0) @binding(0)
var<storage, read> partial_input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> partial_output: array<f32>;

var<workgroup> cache: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let global_i = global_id.x;
    let local_i = local_id.x;
    let n = u32(partial_input[0]);

    if (global_i == 0u) {
        let groups = (n + 63u) / 64u;
        partial_output[0] = f32(groups);
    }

    cache[local_i] = 0.0;

    if(global_i < n) {
        cache[local_i] = partial_input[global_i + 1];
    }

    workgroupBarrier();

    var stride = 32u;
    loop {
        if (local_i < stride) {
            cache[local_i] += cache[local_i + stride];
        }
        workgroupBarrier();

        if (stride == 1u) {
            break;
        }
        stride /= 2u;
    }

    // -----------------------------------
    // Write one partial sum per workgroup
    // -----------------------------------

    if (local_i == 0u && n > 64) {
        partial_output[workgroup_id.x + 1] = cache[0];
    } else if (n <= 64) {
        partial_output[0] = cache[0];
    }
}