@group(0) @binding(0)
var<storage, read> diag: array<f32>;

@group(0) @binding(1)
var<storage, read> off_diag: array<f32>;

@group(0) @binding(2)
var<storage, read> stride: array<u32>;

@group(0) @binding(3)
var<storage, read> p: array<f32>;

@group(0) @binding(4)
var<storage, read_write> ap: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let ndim = arrayLength(&stride);
    let n = arrayLength(&diag);

    if(i >= n) {
        return;
    }

    var sum = diag[i] * p[i];

    for(var axis = 0u; axis < ndim; axis++) {
        // positive neighbour
        let j = i + stride[axis];
        if(j < n) {
            let coeff = off_diag[i * ndim + axis];

            if(coeff != 0.0) {
                sum += coeff * p[j];
            }
        }

        // negative neighbour
        if(i >= stride[axis]) {
            let j = i - stride[axis];
            if(j < n) {
                let coeff_neg = off_diag[j * ndim + axis];

                if(coeff_neg != 0.0) {
                    sum += coeff_neg * p[j];
                }
            }
        }
    }
    ap[i] = sum;
}