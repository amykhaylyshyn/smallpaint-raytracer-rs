use rand::{
    distributions::{Distribution, Uniform},
    prelude::ThreadRng,
};

pub struct Random {
    uniform: Uniform<f64>,
    rng: ThreadRng,
}

impl Random {
    pub fn new() -> Self {
        Self {
            uniform: Uniform::new(0.0, 1.0),
            rng: rand::thread_rng(),
        }
    }

    pub fn sample(&mut self) -> f64 {
        self.uniform.sample(&mut self.rng)
    }
}
