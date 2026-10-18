use macroquad::color::*;

pub struct ColorGradient {
    pub low: Color,
    pub mid: Color,
    pub high: Color,

    pub low_val: f32,
    pub high_val: f32,
}

impl ColorGradient {
    pub fn get(&self, val: f32) -> Color {
        let t = (val - self.low_val) / (self.high_val - self.low_val);
        if t >= 1.0 {
            return self.high;
        }
        if t <= 0.0 {
            return self.low;
        }
        let l = (1.0 - 2.0 * t.abs()).max(0.0);
        let m = (1.0 - 2.0 * (t-0.5).abs()).max(0.0);
        let h = (1.0 - 2.0 * (t-1.0).abs()).max(0.0);
        Color {
            r: l * self.low.r + h * self.high.r + m * self.mid.r,
            g: l * self.low.g + h * self.high.g + m * self.mid.g,
            b: l * self.low.b + h * self.high.b + m * self.mid.b,
            a: l * self.low.a + h * self.high.a + m * self.mid.a
        }
    }
}

pub struct VisualParameter {
    pub fluid_cell: Color,
    pub solid_cell: Color,
    pub smoke_color: Color,

    pub divergence_gradient: ColorGradient,
    pub pressure_gradient: ColorGradient,
    pub velocity_gradient: ColorGradient,
}