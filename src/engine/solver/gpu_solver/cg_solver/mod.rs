use std::time::Instant;

use pollster::block_on;

use crate::engine::solver::{SolvingAlgorithm, linear_operator::PoissonOperator};

use super::gpu::{Kernel, KernelManager};

struct Kernels {
    pub apply: Kernel,
    pub mul: Kernel,
    pub reduce: Kernel,
    pub update_xr: Kernel,
    pub update_p: Kernel
}

struct BindGroups {
    pub apply: wgpu::BindGroup,
    pub dot: wgpu::BindGroup,
    pub reduce: wgpu::BindGroup,
    pub update_xr: wgpu::BindGroup,
    pub error: wgpu::BindGroup,
    pub update_p: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Parameter {
    alpha: f32,
    beta: f32,
}

struct GPUState {
    pub diag: wgpu::Buffer,
    pub off_diag: wgpu::Buffer,
    pub stride: wgpu::Buffer,

    pub x: wgpu::Buffer,
    pub r: wgpu::Buffer,
    pub p: wgpu::Buffer,
    pub ap: wgpu::Buffer,

    pub partial: wgpu::Buffer,

    pub param: wgpu::Buffer,
    pub reduction_size: wgpu::Buffer,
}

pub struct CGSolver {
    kernels: Option<Kernels>,
    gpu_state: Option<GPUState>,
    bind_groups: Option<BindGroups>,
    manager: KernelManager,
}

fn dot(v: &[f32], w: &[f32]) -> f32 {
    v.iter().zip(w.iter()).map(|(x, y)| x * y).sum()
}

fn dot_reduction(a: &[f32], b: &[f32], workgroups: usize) -> Vec<f32> {
    let mut partial_sums: Vec<f32> = vec![0.0; workgroups];
    let n = a.len();
    for w in 0..workgroups {
        let mut cache: [f32; 64] = [0.0; 64];
        for i in 0..64 {
            let global_i = w * 64 + i;
            let local_i = i;

            let mut value = 0.0;
            if global_i < n {
                value = a[global_i] * b[global_i];
            }
            if w == 0 {
                println!("Mul: {}, {} -> {}", a[global_i], b[global_i], value);
            }
            cache[local_i] = value;
        }

        if w == 0 {
            println!("Cache {}: {:?}", w, cache);
        }

        let mut stride = 32;
        loop {
            for i in 0..64 {
                let local_i = i;
                if local_i < stride {
                    cache[local_i] += cache[local_i + stride];
                }
            }
            if stride == 1 {
                break;
            }
            stride /= 2;
        }

        partial_sums[w] = cache[0];
    }
    partial_sums
}

impl SolvingAlgorithm for CGSolver {
    fn new() -> Self {
        let manager = block_on(KernelManager::new());
        
        Self {
            manager: manager,
            kernels: None,
            gpu_state: None,
            bind_groups: None,
        }
    }

    fn init(
        &mut self,
        op: PoissonOperator
    ) {
        self.gpu_state = Some(GPUState::new(&op, &self.manager));
        self.kernels = Some(Kernels::new(op.n, &mut self.manager));
        self.bind_groups = Some(BindGroups::new(
            self.gpu_state.as_ref().expect("No GPU State"), 
            self.kernels.as_ref().expect("No kernels"),
            &mut self.manager
        ));
    }

    fn solve(
        &self,
        x: &mut [f32],
        b: &[f32],
        maxit: usize,
        tol: f32,
    ) -> usize
    {
        assert_eq!(x.len(), b.len());

        let n: usize = x.len();

        x.fill(0.0);

        let buffers = self.gpu_state.as_ref().expect("No GPU State");
        let kernels = self.kernels.as_ref().expect("No kernels");
        let bind_groups = self.bind_groups.as_ref().expect("No bind groups");

        let p = &buffers.p;

        self.manager.update_buffer(&p, b);
        self.manager.update_buffer(&buffers.r, b);
        self.manager.update_buffer(&buffers.x, x);

        let mut operator_time = 0;
        let mut dot_time = 0;
        let mut assign_time = 0;

        let groups = [(n as u32 + 63) / 64, 1, 1];

        let mut rs_old: f32 = self.dot(&bind_groups.error, n as u32).sqrt();
        let scale = dot(b, b).sqrt();

        println!("Check dot: {} and {}", rs_old, scale);

        let mut parameter: Parameter = Parameter { alpha: 0.0, beta: 0.0 };

        for iter in 0..maxit {
            let op_start = Instant::now();
            self.manager.execute_kernel(
                &kernels.apply,
                &bind_groups.apply,
                &groups
            );
            operator_time += op_start.elapsed().subsec_micros();

            let dot_start = Instant::now();
            let denom = self.dot(&bind_groups.dot, n as u32);
            dot_time += dot_start.elapsed().subsec_micros();

            if denom.abs() < 1e-20 {
                println!("CG breakdown");
                return iter;
            }

            parameter.alpha = rs_old / denom;

            let assign_start = Instant::now();
            self.manager.update_uniform(&buffers.param, parameter);
            self.manager.execute_kernel(
                &kernels.update_xr, 
                &bind_groups.update_xr, 
                &groups
            );
            assign_time += assign_start.elapsed().subsec_micros();

            let dot_start = Instant::now();
            let rs_new = self.dot(&bind_groups.error, n as u32);
            dot_time += dot_start.elapsed().subsec_micros();

            let error = rs_new.sqrt() / scale;

            if error < tol {
                println!("Total operator time {:.2?} micros", operator_time);
                println!("Total dot time {:.2?} micros", dot_time);
                println!("Total assign time {:.2?} micros", assign_time);
                self.manager.retrieve_data(
                    &kernels.update_xr, 
                    &buffers.x,
                    x);
                return iter;
            }

            parameter.beta = rs_new / rs_old;

            self.manager.update_uniform(&buffers.param, parameter);

            let assign_start = Instant::now();
            self.manager.execute_kernel(
                &kernels.update_p, 
                &bind_groups.update_p, 
                &groups
            );
            assign_time += assign_start.elapsed().subsec_micros();

            rs_old = rs_new;
        }
        println!("Total operator time {:.2?} micros", operator_time);
        println!("Total dot time {:.2?} micros", dot_time);
        println!("Total assign time {:.2?} micros", assign_time);
        maxit
    }
}

impl CGSolver {
    fn dot(&self, bind_group: &wgpu::BindGroup,initial_size: u32) -> f32 {
        let kernels = self.kernels.as_ref().expect("No kernels");
        let bind_groups = self.bind_groups.as_ref().expect("No bind groups");
        let buffers = self.gpu_state.as_ref().expect("No GPU State");

        self.manager.execute_kernel(
            &kernels.mul,
            bind_group,
            &[(initial_size as u32 + 63) / 64, 1, 1]
        );

        let mut n = initial_size;

        while n > 1 {
            self.manager.update_uniform(&buffers.reduction_size, n);
            self.manager.execute_kernel(
                &kernels.reduce, 
                &bind_groups.reduce, 
                &[(n as u32 + 63) / 64, 1, 1]);

            n = n.div_ceil(64);
        }
        
        let mut result = [0.0; 1];
        self.manager.retrieve_data(
            &kernels.reduce,
            &buffers.partial,
             &mut result);
        result[0]

    }
}

impl BindGroups {
    pub fn new(buffers: &GPUState, kernels: &Kernels, manager: &mut KernelManager) -> Self {
        Self {
            apply: manager.create_bind(&vec![
                    (0, &buffers.diag),
                    (1, &buffers.off_diag),
                    (2, &buffers.stride),
                    (3, &buffers.p),
                    (4, &buffers.ap)
                ], &kernels.apply),
            reduce: manager.create_bind(&vec![
                    (0, &buffers.reduction_size),
                    (1, &buffers.partial)
                ], &kernels.reduce),
            update_xr: manager.create_bind(&vec![
                    (0, &buffers.p),
                    (1, &buffers.ap),
                    (2, &buffers.param),
                    (3, &buffers.x),
                    (4, &buffers.r),
                ], &kernels.update_xr),
            dot: manager.create_bind(&vec![
                    (0, &buffers.p),
                    (1, &buffers.ap),
                    (2, &buffers.partial)
                ], &kernels.mul),
            error: manager.create_bind(&vec![
                    (0, &buffers.r),
                    (1, &buffers.r),
                    (2, &buffers.partial)
                ], &kernels.mul),
            update_p: manager.create_bind(&vec![
                    (0, &buffers.r),
                    (1, &buffers.param),
                    (2, &buffers.p)
                ], &kernels.update_p), 
        }
    }
}

impl Kernels {
    pub fn new(n: usize, manager: &mut KernelManager) -> Self {
        Self {
            apply: manager.new_kernel(include_str!("../../../../../shaders/apply.wgsl"))
                .add_layout_buffer(0, true)
                .add_layout_buffer(1, true)
                .add_layout_buffer(2, true)
                .add_layout_buffer(3, true)
                .add_layout_buffer(4, false)
                .create::<f32>(&n),
            
            mul: manager.new_kernel(include_str!("../../../../../shaders/mul.wgsl"))
                .add_layout_buffer(0, true)
                .add_layout_buffer(1, true)
                .add_layout_buffer(2, false)
                .create::<f32>(&n),

            reduce: manager.new_kernel(include_str!("../../../../../shaders/partial_reduction.wgsl"))
                .add_layout_uniform(0)
                .add_layout_buffer(1, false)
                .create::<f32>(&1),

            update_xr: manager.new_kernel(include_str!("../../../../../shaders/update_xr.wgsl"))
                .add_layout_buffer(0, true)
                .add_layout_buffer(1, true)
                .add_layout_uniform(2)
                .add_layout_buffer(3, false)
                .add_layout_buffer(4, false)
                .create::<f32>(&n),
            update_p: manager.new_kernel(include_str!("../../../../../shaders/update_p.wgsl"))
                .add_layout_buffer(0, true)
                .add_layout_uniform(1)
                .add_layout_buffer(2, false)
                .create::<f32>(&n),
        }
    }
}

impl GPUState {
    pub fn new(op: &PoissonOperator, manager: &KernelManager) -> Self {
        Self {
            diag: manager.create_const_buffer(&op.diag),
            off_diag: manager.create_const_buffer(&op.off_diag),
            stride: manager.create_const_buffer(&op.stride),

            x: manager.create_buffer::<f32>(&op.n, true, true),
            r: manager.create_buffer::<f32>(&op.n, true, false),
            p: manager.create_buffer::<f32>(&op.n, true, false),
            ap: manager.create_buffer::<f32>(&op.n, false, true),

            partial: manager.create_buffer::<f32>(&((op.n as f32 / 64.0).ceil() as usize), false, true),

            param: manager.create_uniform::<Parameter>(true, true),
            reduction_size: manager.create_uniform::<u32>(true, false),
        }
    }
}