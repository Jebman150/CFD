use wgpu::{util::DeviceExt, wgc::command::CopySide::Source};

pub struct Kernel {
    staging_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline
}

impl Kernel {
    pub fn execute(&self, work_size: &[u32; 3], bind_group: &wgpu::BindGroup, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);

            // One invocation per element
            compute_pass.dispatch_workgroups(work_size[0], work_size[1], work_size[2]);
        }

        queue.submit(Some(encoder.finish()));
    }

    pub fn retrieve<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, target: &mut [T], device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            buffer,
            0,
            &self.staging_buffer,
            0,
            self.staging_buffer.size(),
        );

        queue.submit(Some(encoder.finish()));

        let buffer_slice = self.staging_buffer.slice(..);

        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None
        });

        let data = buffer_slice.get_mapped_range();

        target.copy_from_slice(bytemuck::cast_slice(&data));

        drop(data);
        self.staging_buffer.unmap();
    }

    pub fn get_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}

pub struct KernelManager {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,

    shader: Option<wgpu::ShaderModule>,
    buffer_layout: Vec<wgpu::BindGroupLayoutEntry>,
}

impl KernelManager {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find GPU adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("Failed to create device");

        Self {
            instance: instance,
            adapter: adapter,
            device: device,
            queue: queue,

            shader: None,
            buffer_layout: Vec::new(),
        }
    }

    /////////////////////////////
    /// Buffer managing
    /////////////////////////////
    
    pub fn create_buffer<T: bytemuck::Pod>(&self, n: &usize, write: bool, output: bool) -> wgpu::Buffer {
        let buffer_size = (n * std::mem::size_of::<T>()) as wgpu::BufferAddress;

        let mut usage = wgpu::BufferUsages::STORAGE;
        if write {
            usage = usage | wgpu::BufferUsages::COPY_DST;
        }
        if output {
            usage = usage | wgpu::BufferUsages::COPY_SRC;
        }

        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Buffer"),
            size: buffer_size,
            usage: usage,
            mapped_at_creation: false
        })
    }

    pub fn create_const_buffer<T: bytemuck::Pod>(&self, data: &[T]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
        })
    }

    pub fn update_buffer<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, data: &[T]) -> &Self {
        self.queue.write_buffer(
            &buffer,
            0,
            bytemuck::cast_slice(&data));
        self
    }

    /////////////////////////////
    /// Bind group creation
    /////////////////////////////

    pub fn create_bind(&mut self, buffer_list: &Vec<(u32, &wgpu::Buffer)>, kernel: &Kernel) -> wgpu::BindGroup {
        let mut group_entries = Vec::new();

        for buffer in buffer_list.iter() {
            group_entries.push(wgpu::BindGroupEntry {
                binding: buffer.0,
                resource: buffer.1.as_entire_binding(),
            });
        }

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind group"),
            layout: &kernel.get_layout(),
            entries: &group_entries
        })
    }

    /////////////////////
    /// Layout creation
    /////////////////////
    pub fn new_kernel(&mut self, shader_source: &str) -> &mut Self {
        self.shader = Some(self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        }));
        self
    }

    pub fn add_layout_buffer(&mut self, binding: u32, read_only: bool) -> &mut Self {
        self.buffer_layout.push(wgpu::BindGroupLayoutEntry {
            binding: binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        self
    }

    pub fn create<T: bytemuck::Pod>(&mut self, return_size: &usize) -> Kernel {
        let buffer_size = (return_size * std::mem::size_of::<T>()) as wgpu::BufferAddress;

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind group layout"),
            entries: &self.buffer_layout
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        self.buffer_layout = Vec::new();

        Kernel {
            staging_buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Input Buffer"),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::MAP_READ
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            bind_group_layout: bind_group_layout,
            pipeline: self.device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &self.shader.take().expect("No shader for kernel"),
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
        }
    }

    pub fn execute_kernel(&self, kernel: &Kernel, bind_group: &wgpu::BindGroup, work_size: &[u32; 3]) {
        kernel.execute(
            work_size,
            bind_group,
            &self.device,
            &self.queue);
    }

    pub fn retrieve_data<T: bytemuck::Pod>(
        &self,
        kernel: &Kernel,
        buffer: &wgpu::Buffer,
        target: &mut [T]
    ) {
        kernel.retrieve(
            buffer,
            target,
            &self.device,
            &self.queue
        );
    }
}