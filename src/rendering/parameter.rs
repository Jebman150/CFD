use macroquad::color::*;

pub struct ColorGradient {
    pub low: Color,
    pub high: Color,

    pub low_val: f32,
    pub high_val: f32,
}

impl ColorGradient {
    pub fn get(&self, val: f32) -> Color {
        let t = (val - self.low_val) / (self.high_val - self.low_val);
        Color {
            r: (1.0 - t) * self.low.r + t * self.high.r,
            g: (1.0 - t) * self.low.g + t * self.high.g,
            b: (1.0 - t) * self.low.b + t * self.high.b,
            a: (1.0 - t) * self.low.a + t * self.high.a
        }
    }
}

pub struct VisualParameter {
    pub fluid_cell: Color,
    pub solid_cell: Color,
    pub smoke_color: Color,
    pub divergence: ColorGradient,
    pub pressure: ColorGradient,
}