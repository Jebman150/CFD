use std::time::{Duration, Instant};

use pollster::block_on;
use wgpu::{VertexStepMode::Instance, wgc::validation::BindingTypeName::Buffer};

use crate::engine::solver::{SolvingAlgorithm, linear_operator::PoissonOperator};

use super::gpu::{Kernel, KernelManager};

struct Kernels {
    pub apply: Kernel,
    pub reduce: Kernel,
    pub update_xr: Kernel,
    pub update_p: Kernel
}

struct BindGroups {
    pub apply: wgpu::BindGroup,

    pub reduce1: wgpu::BindGroup,
    pub reduce2: wgpu::BindGroup,
    pub reduce_out1: wgpu::BindGroup,
    pub reduce_out2: wgpu::BindGroup,

    pub update_xr: wgpu::BindGroup,
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

    pub partial1: wgpu::Buffer,
    pub partial2: wgpu::Buffer,
    pub dot_out: wgpu::Buffer,

    pub param: wgpu::Buffer,
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
        let parameter: Parameter = Parameter {
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

        let mut encoding_time: u32 = 0;
        let mut submission_time: u32 = 0;
        let mut retrieving_time: u32 = 0;

        let mut encoder = manager.get_encoder();

        let mut encoding_start = Instant::now();
        for iter in 0..maxit {
            kernels.apply.execute(
                &groups, 
                &bind_groups.apply, 
                &mut encoder
            );

            let mut current_size = n as u32;
            let mut current_reduce_group = 
            if n <= 64 {
                &bind_groups.reduce_out1
            }else {
                &bind_groups.reduce1
            };
            while current_size > 1 {
                kernels.reduce.execute(
                    &[(current_size as u32 + 63) / 64, 1, 1],
                    current_reduce_group,
                    &mut encoder
                );

                if current_reduce_group == &bind_groups.reduce_out1 || current_reduce_group == &bind_groups.reduce_out2 {
                    break;
                }

                current_size = current_size.div_ceil(64);

                current_reduce_group =
                if current_reduce_group == &bind_groups.reduce1 {
                    
                    if current_size <= 64 {
                        &bind_groups.reduce_out2
                    } else {
                        &bind_groups.reduce2
                    }
                } else {
                    if current_size <= 64 {
                        &bind_groups.reduce_out1
                    } else {
                        &bind_groups.reduce1
                    }
                };
            }

            kernels.update_xr.execute(
                &groups,
                &bind_groups.update_xr,
                &mut encoder
            );

            let mut current_size = n as u32;
            let mut current_reduce_group = 
            if n <= 64 {
                &bind_groups.reduce_out1
            }else {
                &bind_groups.reduce1
            };
            while current_size > 1 {
                kernels.reduce.execute(
                    &[(current_size as u32 + 63) / 64, 1, 1],
                    current_reduce_group,
                    &mut encoder
                );

                current_size = current_size.div_ceil(64);

                current_reduce_group =
                if current_reduce_group == &bind_groups.reduce1 {
                    if current_size <= 64 {
                        &bind_groups.reduce_out2
                    } else {
                        &bind_groups.reduce2
                    }
                } else {
                    if current_size <= 64 {
                        &bind_groups.reduce_out1
                    } else {
                        &bind_groups.reduce1
                    }
                };
            }


            encoding_time += encoding_start.elapsed().subsec_micros();
            let submission_begin: Instant = Instant::now();
            manager.consume(encoder);
            submission_time += submission_begin.elapsed().subsec_micros();
            encoder = manager.get_encoder();
            encoding_start = Instant::now();

            if iter%10 == 0 && iter != 0 {
                let retrieving_begin = Instant::now();
                let mut error: Vec<f32> = vec![0.0];
                manager.retrieve_data(
                    &kernels.reduce, 
                    &buffers.dot_out, 
                    &mut error);
                //println!("Error readout: {}", error[0].sqrt()/scale);
                retrieving_time += retrieving_begin.elapsed().subsec_micros();

                if error[0].sqrt()/scale < tol {
                    let total_retrieve = Instant::now();
                    manager.retrieve_data(
                        &kernels.update_xr, 
                        &buffers.x,
                        x);

                    println!("Encoding time: {}", encoding_time);
                    println!("Submission time: {}", submission_time);
                    println!("Retrieving time: {}", retrieving_time);
                    //println!("End result retrieving time: {}", total_retrieve.elapsed().subsec_micros());
                    return iter;
                }

            }

            kernels.update_p.execute(
                &groups,
                &bind_groups.update_p, 
                &mut encoder
            );
        }
        manager.consume(encoder);
        manager.retrieve_data(
            &kernels.update_xr, 
            &buffers.x,
            x);
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
                    (4, &buffers.ap),
                    (5, &buffers.partial1)
                ], &kernels.apply),
            reduce1: manager.create_bind(&vec![
                    (0, &buffers.partial1),
                    (1, &buffers.partial2)
                ], &kernels.reduce),
            reduce2: manager.create_bind(&vec![
                    (0, &buffers.partial2),
                    (1, &buffers.partial1)
                ], &kernels.reduce),
            reduce_out1: manager.create_bind(&vec![
                    (0, &buffers.partial1),
                    (1, &buffers.dot_out)
                ], &kernels.reduce),
            reduce_out2: manager.create_bind(&vec![
                    (0, &buffers.partial2),
                    (1, &buffers.dot_out)
                ], &kernels.reduce),
            update_xr: manager.create_bind(&vec![
                    (0, &buffers.p),
                    (1, &buffers.ap),
                    (2, &buffers.dot_out),
                    (3, &buffers.param),
                    (4, &buffers.x),
                    (5, &buffers.r),
                    (6, &buffers.partial1),
                ], &kernels.update_xr),

            update_p: manager.create_bind(&vec![
                    (0, &buffers.r),
                    (1, &buffers.dot_out),
                    (2, &buffers.param),
                    (3, &buffers.p)
                ], &kernels.update_p)
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
                .add_layout_buffer(5, false)
                .create::<f32>(&(n+1)),

            reduce: manager.new_kernel(include_str!("../../../../../shaders/partial_reduction.wgsl"))
                .add_layout_buffer(0, true)
                .add_layout_buffer(1, false)
                .create::<f32>(&(n+1)),

            update_xr: manager.new_kernel(include_str!("../../../../../shaders/update_xr.wgsl"))
                .add_layout_buffer(0, true)
                .add_layout_buffer(1, true)
                .add_layout_buffer(2, true)
                .add_layout_buffer(3, true)
                .add_layout_buffer(4, false)
                .add_layout_buffer(5, false)
                .add_layout_buffer(6, false)
                .create::<f32>(&n),

            update_p: manager.new_kernel(include_str!("../../../../../shaders/update_p.wgsl"))
                .add_layout_buffer(0, true)
                .add_layout_buffer(1, true)
                .add_layout_buffer(2, false)
                .add_layout_buffer(3, false)
                .create::<f32>(&(n+1))
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

            partial1: manager.create_buffer::<f32>(&(op.n+1), false, true),
            partial2: manager.create_buffer::<f32>(&(op.n+1), false, true),
            dot_out: manager.create_buffer::<f32>(&1, false, true),

            param: manager.create_buffer::<Parameter>(&1, true, false),
        }
    }
}