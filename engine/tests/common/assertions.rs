

pub fn assert_approximately_equal(
    log_name: &str,
    actual: &[f32],
    expected: &[f32],
    abs_tol: f32,
    rel_tol: f32,
) {
    assert_eq!(actual.len(), expected.len());

    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let tolerance = abs_tol + rel_tol * e.abs();

        if (a - e).abs() > tolerance {
            write_error_diagnostics(log_name, actual, expected, abs_tol, rel_tol);
        }
        assert!(
            (a - e).abs() <= tolerance,
            "Mismatch at index {i}: GPU={a}, CPU={e}, error={}, tolerance={tolerance}",
            (a - e).abs(),
        );
    }
}


use plotters::prelude::*;

pub fn write_error_diagnostics(
    log_name: &str,
    actual: &[f32],
    expected: &[f32],
    abs_tol: f32,
    rel_tol: f32,
) {
    if actual.len() != expected.len() || actual.is_empty() {
        eprintln!("Invalid input: actual and expected must have equal, non-zero lengths");
        return;
    }

    if !abs_tol.is_finite()
        || !rel_tol.is_finite()
        || abs_tol < 0.0
        || rel_tol < 0.0
    {
        eprintln!("Invalid tolerances");
        return;
    }

    let mut samples = Vec::new();

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if !a.is_finite() || !e.is_finite() {
            continue;
        }

        let error = a - e;
        let abs_error = error.abs();
        let tolerance = abs_tol + rel_tol * e.abs();

        samples.push((i, a, e, error, abs_error, tolerance));
    }

    if samples.is_empty() {
        eprintln!("No finite samples to plot");
        return;
    }

    let errors: Vec<f32> = samples.iter().map(|s| s.3).collect();
    let abs_errors: Vec<f32> = samples.iter().map(|s| s.4).collect();

    let n = samples.len();

    let mean = errors.iter().sum::<f32>() / n as f32;
    let mae = abs_errors.iter().sum::<f32>() / n as f32;
    let rmse = (errors.iter().map(|e| e * e).sum::<f32>() / n as f32).sqrt();
    let max_error = abs_errors.iter().copied().fold(0.0_f32, f32::max);

    let passed = samples.iter().filter(|s| s.4 <= s.5).count();
    let pass_rate = passed as f32 / n as f32 * 100.0;

    println!("Error diagnostics:");
    println!("  Samples:   {n}");
    println!("  Mean:      {mean:.6}");
    println!("  MAE:       {mae:.6}");
    println!("  RMSE:      {rmse:.6}");
    println!("  Max error: {max_error:.6}");
    println!("  Pass rate: {pass_rate:.2}%");

    let file_name = format!("error_diagnostics_{}.png", log_name);
    let root = BitMapBackend::new(&file_name, (1400, 1000))
        .into_drawing_area();

    root.fill(&WHITE).unwrap();

    let areas = root.split_evenly((2, 2));

    // Helper for robust plot ranges.
    fn padded_range(min: f32, max: f32) -> (f32, f32) {
        let span = max - min;
        let padding = if span > 0.0 {
            span * 0.1
        } else {
            min.abs().max(1.0) * 0.1
        };
        (min - padding, max + padding)
    }

    // ---------------------------------------------------------
    // 1. Signed error distribution (histogram)
    // ---------------------------------------------------------

    {
        let min = errors.iter().copied().fold(f32::INFINITY, f32::min);
        let max = errors.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let (min, max) = padded_range(min, max);
        let bins = 50usize;
        let bin_width = (max - min) / bins as f32;

        let mut counts = vec![0usize; bins];

        for &error in &errors {
            let idx = (((error - min) / bin_width) as usize).min(bins - 1);
            counts[idx] += 1;
        }

        let max_count = *counts.iter().max().unwrap_or(&1);

        let mut chart = ChartBuilder::on(&areas[0])
            .caption("Signed Error Distribution", ("sans-serif", 25))
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(min..max, 0usize..(max_count + 1))
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("Error (actual - expected)")
            .y_desc("Frequency")
            .draw()
            .unwrap();

        chart
            .draw_series(counts.iter().enumerate().map(|(i, &count)| {
                let x0 = min + i as f32 * bin_width;
                let x1 = x0 + bin_width;

                Rectangle::new(
                    [(x0, 0), (x1, count)],
                    BLUE.mix(0.65).filled(),
                )
            }))
            .unwrap();

        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(0.0, 0), (0.0, max_count + 1)],
                RED.stroke_width(2),
            )))
            .unwrap()
            .label("Zero error")
            .legend(|(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2))
            });

        chart.configure_series_labels().draw().unwrap();
    }

    // ---------------------------------------------------------
    // 2. Absolute error distribution
    // ---------------------------------------------------------

    {
        let max = abs_errors.iter().copied().fold(0.0_f32, f32::max);
        let max = max.max(1e-8) * 1.05;

        let bins = 50usize;
        let bin_width = max / bins as f32;
        let mut counts = vec![0usize; bins];

        for &error in &abs_errors {
            let idx = ((error / bin_width) as usize).min(bins - 1);
            counts[idx] += 1;
        }

        let max_count = *counts.iter().max().unwrap_or(&1);

        let mut chart = ChartBuilder::on(&areas[1])
            .caption("Absolute Error Distribution", ("sans-serif", 25))
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0f32..max, 0usize..(max_count + 1))
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("Absolute error")
            .y_desc("Frequency")
            .draw()
            .unwrap();

        chart
            .draw_series(counts.iter().enumerate().map(|(i, &count)| {
                let x0 = i as f32 * bin_width;
                let x1 = x0 + bin_width;

                Rectangle::new(
                    [(x0, 0), (x1, count)],
                    GREEN.mix(0.65).filled(),
                )
            }))
            .unwrap();

        // A single tolerance line is only valid for constant tolerances.
        // Here the effective tolerance varies with expected.
        let max_tol = samples
            .iter()
            .map(|s| s.5)
            .fold(0.0_f32, f32::max);

        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(abs_tol, 0), (abs_tol, max_count + 1)],
                RED.stroke_width(2),
            )))
            .unwrap()
            .label("Absolute tolerance component")
            .legend(|(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2))
            });

        chart.configure_series_labels().draw().unwrap();

        let _ = max_tol;
    }

    // ---------------------------------------------------------
    // 3. Actual vs expected
    // ---------------------------------------------------------

    {
        let min = samples
            .iter()
            .map(|s| s.1.min(s.2))
            .fold(f32::INFINITY, f32::min);

        let max = samples
            .iter()
            .map(|s| s.1.max(s.2))
            .fold(f32::NEG_INFINITY, f32::max);

        let (min, max) = padded_range(min, max);

        let mut chart = ChartBuilder::on(&areas[2])
            .caption("Actual vs Expected", ("sans-serif", 25))
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(min..max, min..max)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("Expected")
            .y_desc("Actual")
            .draw()
            .unwrap();

        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(min, min), (max, max)],
                RED.stroke_width(2),
            )))
            .unwrap()
            .label("Ideal")
            .legend(|(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2))
            });

        chart
            .draw_series(samples.iter().map(|s| {
                Circle::new(
                    (s.2, s.1),
                    2,
                    if s.4 <= s.5 {
                        GREEN.mix(0.5).filled()
                    } else {
                        RED.mix(0.7).filled()
                    },
                )
            }))
            .unwrap()
            .label("Samples");

        chart.configure_series_labels().draw().unwrap();
    }

    // ---------------------------------------------------------
    // 4. Error over sample index
    // ---------------------------------------------------------

    {
        let min = errors.iter().copied().fold(f32::INFINITY, f32::min);
        let max = errors.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let (min, max) = padded_range(min.min(0.0), max.max(0.0));

        let mut chart = ChartBuilder::on(&areas[3])
            .caption("Error Over Sample Index", ("sans-serif", 25))
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                0usize..actual.len().max(2),
                min..max,
            )
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("Sample index")
            .y_desc("Signed error")
            .draw()
            .unwrap();

        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(0, 0.0), (actual.len(), 0.0)],
                BLACK.stroke_width(1),
            )))
            .unwrap();

        chart
            .draw_series(samples.iter().map(|s| {
                Circle::new(
                    (s.0, s.3),
                    2,
                    if s.4 <= s.5 {
                        GREEN.filled()
                    } else {
                        RED.filled()
                    },
                )
            }))
            .unwrap();
    }

    root.present().unwrap();

    println!("  Plot saved to error_diagnostics.png");
}
