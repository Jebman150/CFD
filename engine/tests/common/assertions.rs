

pub fn assert_approximately_equal(
    actual: &[f32],
    expected: &[f32],
    abs_tol: f32,
    rel_tol: f32,
) {
    assert_eq!(actual.len(), expected.len());

    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let tolerance = abs_tol + rel_tol * e.abs();

        assert!(
            (a - e).abs() <= tolerance,
            "Mismatch at index {i}: GPU={a}, CPU={e}, error={}, tolerance={tolerance}",
            (a - e).abs(),
        );
    }
}