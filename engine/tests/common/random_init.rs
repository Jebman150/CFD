use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub fn generate_random_mac_velocity(
    size: [usize; 2],
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let [nx, ny] = size;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let u = (0..(nx + 1) * ny)
        .map(|_| rng.random_range(-10.0..10.0))
        .collect();

    let v = (0..nx * (ny + 1))
        .map(|_| rng.random_range(-10.0..10.0))
        .collect();

    (u, v)
}

pub fn generate_random_scalar_field(
    size: [usize; 2],
    seed: u64,
) -> Vec<f32> {
    let [nx, ny] = size;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let f = (0..(nx * ny))
        .map(|_| rng.random_range(-10.0..10.0))
        .collect();

    f
}