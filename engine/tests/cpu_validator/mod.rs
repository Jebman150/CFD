

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


fn bilerp(
    v00: f32,
    v10: f32,
    v01: f32,
    v11: f32,
    tx: f32,
    ty: f32,
) -> f32 {
    let bottom = v00 + tx * (v10 - v00);
    let top = v01 + tx * (v11 - v01);
    bottom + ty * (top - bottom)
}

fn sample_mac_x(
    field: &[f32],
    pos_x: f32,
    pos_y: f32,
    nx: usize,
    ny: usize,
    dx: f32,
    dy: f32,
) -> f32 {
    // X component is located at:
    // (i * dx, (j + 0.5) * dy)
    let gx = (pos_x / dx).clamp(0.0, nx as f32);
    let gy = (pos_y / dy - 0.5).clamp(0.0, (ny - 1) as f32);

    let x0 = gx.floor() as usize;
    let y0 = gy.floor() as usize;

    let x1 = (x0 + 1).min(nx);
    let y1 = (y0 + 1).min(ny - 1);

    let tx = gx - x0 as f32;
    let ty = gy - y0 as f32;

    // X component has nx + 1 entries per row.
    let stride = nx + 1;

    let v00 = field[y0 * stride + x0];
    let v10 = field[y0 * stride + x1];
    let v01 = field[y1 * stride + x0];
    let v11 = field[y1 * stride + x1];

    bilerp(v00, v10, v01, v11, tx, ty)
}

fn sample_mac_y(
    field: &[f32],
    pos_x: f32,
    pos_y: f32,
    nx: usize,
    ny: usize,
    dx: f32,
    dy: f32,
) -> f32 {
    // Y component is located at:
    // ((i + 0.5) * dx, j * dy)
    let gx = (pos_x / dx - 0.5).clamp(0.0, (nx - 1) as f32);
    let gy = (pos_y / dy).clamp(0.0, ny as f32);

    let x0 = gx.floor() as usize;
    let y0 = gy.floor() as usize;

    let x1 = (x0 + 1).min(nx - 1);
    let y1 = (y0 + 1).min(ny);

    let tx = gx - x0 as f32;
    let ty = gy - y0 as f32;

    // Y component has nx entries per row.
    let stride = nx;

    let v00 = field[y0 * stride + x0];
    let v10 = field[y0 * stride + x1];
    let v01 = field[y1 * stride + x0];
    let v11 = field[y1 * stride + x1];

    bilerp(v00, v10, v01, v11, tx, ty)
}

/// CPU reference implementation of semi-Lagrangian vector advection.
///
/// All fields use MAC staggered-grid storage:
///
/// - velocity_x:       (nx + 1) * ny
/// - velocity_y:       nx * (ny + 1)
/// - vector_field_in_x: (nx + 1) * ny
/// - vector_field_in_y: nx * (ny + 1)
///
/// Returns:
/// - Advected X component: (nx + 1) * ny
/// - Advected Y component: nx * (ny + 1)
pub fn vector_advection_cpu(
    velocity_x: &[f32],
    velocity_y: &[f32],
    vector_field_in_x: &[f32],
    vector_field_in_y: &[f32],
    grid_size: [usize; 2],
    dx: f32,
    dy: f32,
    dt: f32,
) -> (Vec<f32>, Vec<f32>) {
    let [nx, ny] = grid_size;

    assert!(nx > 0 && ny > 0);
    assert!(dx.is_finite() && dx > 0.0);
    assert!(dy.is_finite() && dy > 0.0);
    assert!(dt.is_finite());

    let x_face_count = (nx + 1) * ny;
    let y_face_count = nx * (ny + 1);

    assert_eq!(velocity_x.len(), x_face_count);
    assert_eq!(velocity_y.len(), y_face_count);
    assert_eq!(vector_field_in_x.len(), x_face_count);
    assert_eq!(vector_field_in_y.len(), y_face_count);

    let mut out_x = vec![0.0; x_face_count];
    let mut out_y = vec![0.0; y_face_count];

    // Advect X component, located on X faces.
    for y in 0..ny {
        for x in 0..=nx {
            let i = y * (nx + 1) + x;

            // World position of this X face.
            let pos_x = x as f32 * dx;
            let pos_y = (y as f32 + 0.5) * dy;

            // Sample velocity at the current position.
            let vel_x = sample_mac_x(
                velocity_x,
                pos_x,
                pos_y,
                nx,
                ny,
                dx,
                dy,
            );

            let vel_y = sample_mac_y(
                velocity_y,
                pos_x,
                pos_y,
                nx,
                ny,
                dx,
                dy,
            );

            // Semi-Lagrangian backward tracing.
            let prev_x = pos_x - dt * vel_x;
            let prev_y = pos_y - dt * vel_y;

            // Sample the original vector field at the departure point.
            out_x[i] = sample_mac_x(
                vector_field_in_x,
                prev_x,
                prev_y,
                nx,
                ny,
                dx,
                dy,
            );
        }
    }

    // Advect Y component, located on Y faces.
    for y in 0..=ny {
        for x in 0..nx {
            let i = y * nx + x;

            // World position of this Y face.
            let pos_x = (x as f32 + 0.5) * dx;
            let pos_y = y as f32 * dy;

            // Sample velocity at the current position.
            let vel_x = sample_mac_x(
                velocity_x,
                pos_x,
                pos_y,
                nx,
                ny,
                dx,
                dy,
            );

            let vel_y = sample_mac_y(
                velocity_y,
                pos_x,
                pos_y,
                nx,
                ny,
                dx,
                dy,
            );

            // Semi-Lagrangian backward tracing.
            let prev_x = pos_x - dt * vel_x;
            let prev_y = pos_y - dt * vel_y;

            // Sample the original vector field at the departure point.
            out_y[i] = sample_mac_y(
                vector_field_in_y,
                prev_x,
                prev_y,
                nx,
                ny,
                dx,
                dy,
            );
        }
    }

    (out_x, out_y)
}
