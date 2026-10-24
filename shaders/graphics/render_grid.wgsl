struct Viewport {
    position: vec2<f32>,
    size: vec2<f32>,
};

struct ColorGradient {
    low_color: vec4<f32>,
    mid_color: vec4<f32>,
    high_color: vec4<f32>,

    low_value: f32,
    high_value: f32,
    _padding: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> viewport: Viewport;

@group(0) @binding(1)
var<uniform> active_color_gradient: ColorGradient;

@group(0) @binding(2)
var<storage, read> scalar_values: array<f32>;

struct VertexInput {
    @location(0) local_position: vec2<f32>,
    @location(1) vertex_color: vec4<f32>,
    @location(2) instance_position: vec2<f32>,
    @location(3) instance_rotation: f32,
    @location(4) instance_scale: vec2<f32>,
    @location(5) instance_color: vec4<f32>,
    @location(6) grid_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

fn mix(t: f32, color_low: vec4<f32>, color_high: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        (1.0 - t) * color_low[0] + t * color_high[0],
        (1.0 - t) * color_low[1] + t * color_high[1],
        (1.0 - t) * color_low[2] + t * color_high[2],
        (1.0 - t) * color_low[3] + t * color_high[3],
    );
}

fn lerp_color(t: f32, color_low: vec4<f32>, color_mid: vec4<f32>, color_high: vec4<f32>) -> vec4<f32> {
    let u = clamp(t, 0.0, 1.0);

    if u < 0.5 {
        return mix(u*2, color_low, color_mid);
    }
    return mix((u-0.5)*2, color_mid, color_high);
}

@vertex
fn vs_main(
    input: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    let x = 2.0 * (input.local_position.x + input.instance_position.x) / (viewport.size.x/2.0) - 1.0;
    let y = 1.0 - 2.0 * (input.local_position.y + input.instance_position.y) / (viewport.size.y/2.0);

    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    let value = scalar_values[input.grid_index];
    let t = f32(value - active_color_gradient.low_value) / f32(active_color_gradient.high_value - active_color_gradient.low_value);
    out.color = lerp_color(
        t,
        active_color_gradient.low_color,
        active_color_gradient.mid_color,
        active_color_gradient.high_color
    );

    return out;
}

@fragment
fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}