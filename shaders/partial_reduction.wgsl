@group(0) @binding(0)
var<uniform> n: u32;

@group(0) @binding(1)
var<storage, read_write> partial_sums: array<f32>;

var<workgroup> cache: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let global_i = global_id.x;
    let local_i = local_id.x;
    cache[local_i] = 0.0;

    if(global_i < n) {
        cache[local_i] = partial_sums[global_i];
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

    if (local_i == 0u) {
        partial_sums[workgroup_id.x] = cache[0];
    }
}