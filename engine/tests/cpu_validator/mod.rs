

pub fn divergence_cpu(
    u: &[f32],
    v: &[f32],
    size: [usize; 2],
    dx: f32,
    dy: f32,
) -> Vec<f32> {
    let [nx, ny] = size;

    let mut result = vec![0.0; nx * ny];

    for j in 0..ny {
        for i in 0..nx {
            let u_left = u[j * (nx + 1) + i];
            let u_right = u[j * (nx + 1) + i + 1];

            let v_bottom = v[j * nx + i];
            let v_top = v[(j + 1) * nx + i];

            result[j * nx + i] =
                (u_right - u_left) / dx
                + (v_top - v_bottom) / dy;
        }
    }

    result
}


pub fn scalar_advection_cpu(
    u: &[f32],
    v: &[f32],
    scalar_values: &[f32],
    grid_size: [usize; 2],
    dx: f32,
    dy: f32,
    dt: f32,
) -> Vec<f32> {
    let [nx, ny] = grid_size;

    assert!(nx > 0 && ny > 0);
    assert!(dx > 0.0 && dy > 0.0);
    assert_eq!(u.len(), (nx + 1) * ny);
    assert_eq!(v.len(), nx * (ny + 1));
    assert_eq!(scalar_values.len(), nx * ny);

    // Bilinear interpolation on a grid whose sample points are
    // located at (i + offset_x, j + offset_y).
    fn bilerp_grid(
        values: &[f32],
        width: usize,
        height: usize,
        gx: f32,
        gy: f32,
    ) -> f32 {
        let gx = gx.clamp(0.0, (width - 1) as f32);
        let gy = gy.clamp(0.0, (height - 1) as f32);

        let x0 = gx.floor() as usize;
        let y0 = gy.floor() as usize;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);

        let tx = gx - x0 as f32;
        let ty = gy - y0 as f32;

        let v00 = values[y0 * width + x0];
        let v10 = values[y0 * width + x1];
        let v01 = values[y1 * width + x0];
        let v11 = values[y1 * width + x1];

        let bottom = v00 * (1.0 - tx) + v10 * tx;
        let top = v01 * (1.0 - tx) + v11 * tx;

        bottom * (1.0 - ty) + top * ty
    }

    // Sample scalar at world-space position.
    fn sample_scalar(
        values: &[f32],
        nx: usize,
        ny: usize,
        pos_x: f32,
        pos_y: f32,
        dx: f32,
        dy: f32,
    ) -> f32 {
        let gx = pos_x / dx - 0.5;
        let gy = pos_y / dy - 0.5;

        bilerp_grid(values, nx, ny, gx, gy)
    }

    // U is located at (i * dx, (j + 0.5) * dy).
    fn sample_velocity_x(
        u: &[f32],
        nx: usize,
        ny: usize,
        pos_x: f32,
        pos_y: f32,
        dx: f32,
        dy: f32,
    ) -> f32 {
        let gx = pos_x / dx;
        let gy = pos_y / dy - 0.5;

        bilerp_grid(u, nx + 1, ny, gx, gy)
    }

    // V is located at ((i + 0.5) * dx, j * dy).
    fn sample_velocity_y(
        v: &[f32],
        nx: usize,
        ny: usize,
        pos_x: f32,
        pos_y: f32,
        dx: f32,
        dy: f32,
    ) -> f32 {
        let gx = pos_x / dx - 0.5;
        let gy = pos_y / dy;

        bilerp_grid(v, nx, ny + 1, gx, gy)
    }

    let mut result = vec![0.0; nx * ny];

    for j in 0..ny {
        for i in 0..nx {
            let index = j * nx + i;

            // Scalar cell center.
            let pos_x = (i as f32 + 0.5) * dx;
            let pos_y = (j as f32 + 0.5) * dy;

            // Sample MAC velocity at scalar cell center.
            let vel_x =
                sample_velocity_x(u, nx, ny, pos_x, pos_y, dx, dy);
            let vel_y =
                sample_velocity_y(v, nx, ny, pos_x, pos_y, dx, dy);

            // Semi-Lagrangian backtracing.
            let prev_x = pos_x - dt * vel_x;
            let prev_y = pos_y - dt * vel_y;

            // Sample the previous scalar field.
            result[index] = sample_scalar(
                scalar_values,
                nx,
                ny,
                prev_x,
                prev_y,
                dx,
                dy,
            );
        }
    }

    result
}
