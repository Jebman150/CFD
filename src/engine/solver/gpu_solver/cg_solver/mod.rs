use std::time::Instant;

use pollster::block_on;
use wgpu::wgc::validation::BindingTypeName::Buffer;

use crate::engine::solver::{SolvingAlgorithm, linear_operator::PoissonOperator};

use super::gpu::{Kernel, KernelManager};

struct Kernels {
    pub update_alpha: Kernel,
    pub update_beta: Kernel,
    pub apply: Kernel,
    pub mul: Kernel,
    pub reduce: Kernel,
    pub update_xr: Kernel,
    pub update_p: Kernel
}

struct BindGroups {
    pub update_alpha: wgpu::BindGroup,
    pub update_beta: wgpu::BindGroup,
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
    rs: f32,
    pad: f32
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

fn mul(v: &[f32], w: &[f32]) -> Vec<f32> {
    v.iter().zip(w.iter()).map(|(x, y)| x * y).collect()
}

fn reduce(p: &mut [f32], n: usize, workgroups: usize) {
    for w in 0..workgroups {
        let mut cache: [f32; 64] = [0.0; 64];
        for i in 0..64 {
            let global_i = w * 64 + i;
            let local_i = i;

            if global_i >= n {
                continue;
            }

            cache[local_i] = p[global_i];
        }

        let mut stride = 32;
        loop {
            for i in 0..64 {
                let global_i = w * 64 + i;
                let local_i = i;

                if global_i >= n {
                    continue;
                }

                if local_i < stride {
                    cache[local_i] += cache[local_i + stride];
                }
            }
            if stride == 1 {
                break;
            }
            stride /= 2;
        }

        p[w] = cache[0];
    }
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
        let manager = &self.manager;

        manager.update_buffer(&buffers.p, b);
        manager.update_buffer(&buffers.r, b);
        manager.update_buffer(&buffers.x, x);

        let mut scale = dot(b, b);
        let mut parameter: Parameter = Parameter {
            alpha: 0.0,
            beta: 0.0,
            rs: scale,
            pad: 0.0,
        };
        manager.update_buffer(
            &buffers.param, 
            &[parameter]
        );

        scale = scale.sqrt();

        let groups = [(n as u32 + 63) / 64, 1, 1];

        let mut encoder = manager.get_encoder();
        for iter in 0..maxit {
            kernels.apply.execute(
                &groups, 
                &bind_groups.apply, 
                &mut encoder
            );

            kernels.mul.execute(
                &[(n as u32 + 63) / 64, 1, 1],
                &bind_groups.dot,
                &mut encoder
            );

            let mut current_size = n as u32;
            while current_size > 1 {
                self.manager.update_uniform(&buffers.reduction_size, current_size);
                kernels.reduce.execute(
                    &[(current_size as u32 + 63) / 64, 1, 1],
                    &bind_groups.reduce,
                    &mut encoder
                );

                self.manager.consume(encoder);
                encoder = self.manager.get_encoder();

                current_size = current_size.div_ceil(64);
            }

            kernels.update_alpha.execute(
                &[1, 1, 1],
                &bind_groups.update_alpha,
                &mut encoder
            );

            kernels.update_xr.execute(
                &groups,
                &bind_groups.update_xr,
                &mut encoder
            );

            kernels.mul.execute(
                &[(n as u32 + 63) / 64, 1, 1],
                &bind_groups.error,
                &mut encoder
            );

            let mut current_size = n as u32;
            while current_size > 1 {
                self.manager.update_uniform(&buffers.reduction_size, current_size);
                kernels.reduce.execute(
                    &[(current_size as u32 + 63) / 64, 1, 1],
                    &bind_groups.reduce,
                    &mut encoder
                );

                self.manager.consume(encoder);
                encoder = self.manager.get_encoder();

                current_size = current_size.div_ceil(64);
            }

            if iter%10 == 0 {
                manager.consume(encoder);
                let mut error: Vec<f32> = vec![0.0];
                manager.retrieve_data(
                    &kernels.reduce, 
                    &buffers.partial, 
                    &mut error);
                //println!("Error readout: {}", error[0].sqrt()/scale);
                if error[0].sqrt()/scale < tol {
                    manager.retrieve_data(
                        &kernels.update_xr, 
                        &buffers.x,
                        x);
                    return iter;
                }
                encoder = manager.get_encoder();
            }

            kernels.update_beta.execute(
                &[1, 1, 1], 
                &bind_groups.update_beta, 
                &mut encoder
            );

            kernels.update_p.execute(
                &groups,
                &bind_groups.update_p, 
                &mut encoder
            );
        }
        maxit
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

            update_alpha: manager.create_bind(&vec![
                    (0, &buffers.partial),
                    (1, &buffers.param)
            ], &kernels.update_alpha),

            update_beta: manager.create_bind(&vec![
                    (0, &buffers.partial),
                    (1, &buffers.param)
            ], &kernels.update_beta),
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
                .add_layout_buffer(2, true)
                .add_layout_buffer(3, false)
                .add_layout_buffer(4, false)
                .create::<f32>(&n),

            update_p: manager.new_kernel(include_str!("../../../../../shaders/update_p.wgsl"))
                .add_layout_buffer(0, true)
                .add_layout_buffer(1, true)
                .add_layout_buffer(2, false)
                .create::<f32>(&n),

            update_alpha: manager.new_kernel(include_str!("../../../../../shaders/update_alpha.wgsl"))
                .add_layout_buffer(0, true)
                .add_layout_buffer(1, false)
                .create::<f32>(&1),

            update_beta:  manager.new_kernel(include_str!("../../../../../shaders/update_beta.wgsl"))
                .add_layout_buffer(0, true)
                .add_layout_buffer(1, false)
                .create::<f32>(&1)
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
            r: manager.create_buffer::<f32>(&op.n, true, true),
            p: manager.create_buffer::<f32>(&op.n, true, true),
            ap: manager.create_buffer::<f32>(&op.n, false, true),

            partial: manager.create_buffer::<f32>(&op.n, false, true),

            param: manager.create_buffer::<Parameter>(&1, true, false),
            reduction_size: manager.create_uniform::<u32>(true, false),
        }
    }
}