struct Parameter {
    grid_size: vec2<u32>,
    dx: vec2<f32>,
    density: f32,
    dt: f32,
};

@group(0) @binding(0)
var<storage, read> is_solid_cell: array<u32>;   // 0 => false, !0 => true

@group(0) @binding(1)
var<storage, read_write> velocity_x: array<f32>;

@group(0) @binding(2)
var<storage, read_write> velocity_y: array<f32>;

@group(0) @binding(3)
var<uniform> params: Parameter;

@compute @workgroup_size(64)
fn apply_solid_cells(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let index = global_id.x;
    let nx = params.grid_size.x;
    let ny = params.grid_size.y;
    let cell_count = nx * ny;

    if (index >= cell_count) {
        return;
    }

    if (is_solid_cell[index] == 0) {
        return;
    }
    
    let cell_x = index % nx;
    let cell_y = index / nx;

    velocity_x[cell_y * (nx + 1) + cell_x] = 0.0;
    velocity_x[cell_y * (nx + 1) + cell_x + 1] = 0.0;

    velocity_y[cell_y * nx + cell_x] = 0.0;
    velocity_y[(cell_y + 1) * nx + cell_x] = 0.0;
}