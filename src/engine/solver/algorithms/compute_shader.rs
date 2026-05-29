use std::io::pipe;

use wgpu::PipelineCompilationOptions;
use wgpu::util::DeviceExt;

pub struct ComputeShader {
    name: String,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,

    shader: wgpu::ShaderModule,

    input_buffers: Vec<(wgpu::Buffer, u32)>,
    output_buffers: Vec<(wgpu::Buffer, u32)>,
    staging_buffers: Vec<wgpu::Buffer>,
    output_buffers_sizes: Vec<u64>,

    bind_group: Option<wgpu::BindGroup>,
    pipeline: Vec<(String, wgpu::ComputePipeline)>,
}

impl ComputeShader {
    pub async fn new(source_path: &str) -> Self {
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(source_path.into()),
        });

        Self {
            name: source_path.to_owned(),
            instance: instance,
            adapter: adapter,
            device: device,
            queue: queue,

            shader: shader,

            input_buffers: Vec::new(),
            output_buffers: Vec::new(),
            staging_buffers: Vec::new(),
            output_buffers_sizes: Vec::new(),

            bind_group: None,
            pipeline: Vec::new(),
        }
    }

    pub fn update_input_buffer<T: bytemuck::Pod>(&self, binding: u32, data: &[T]) -> &Self {
        let buffer = &self.input_buffers
            .iter()
            .find(|b| b.1 == binding)
            .expect("Invalid binding - could not set input buffer at binding").0;

        self.queue.write_buffer(
            &buffer,
            0,
            bytemuck::cast_slice(&data));
        self
    }

    pub fn add_buffer<T: bytemuck::Pod>(&mut self, binding: u32, data: &[T], updatable: bool) -> &mut Self {
        let mut usage = wgpu::BufferUsages::STORAGE;
        if updatable {
            usage = usage | wgpu::BufferUsages::COPY_DST;
        }

        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(&data),
            usage: usage,
        });
        self.input_buffers.push((buffer, binding));

        self
    }

    pub fn add_empty_buffer<T: bytemuck::Pod>(&mut self, binding: u32, n: usize) -> &mut Self {
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let buffer_size = (n * std::mem::size_of::<T>()) as wgpu::BufferAddress;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Buffer"),
            size: buffer_size,
            usage: usage,
            mapped_at_creation: false
        });
        self.input_buffers.push((buffer, binding));

        self
    }

    pub fn add_output_buffer<T>(&mut self, binding: u32, n: usize) -> &mut Self {
        let buffer_size = (n * std::mem::size_of::<T>()) as wgpu::BufferAddress;
        self.output_buffers_sizes.push(buffer_size);

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.staging_buffers.push(staging_buffer);

        let gpu_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.output_buffers.push((gpu_buffer, binding));

        self
    }

    pub fn create(&mut self, entry_points: &[String]) {
        let mut buffer_list = Vec::new();
        let mut buffer_layout = Vec::new();
        for buffer in self.input_buffers.iter() {
            buffer_list.push(wgpu::BindGroupEntry {
                binding: buffer.1,
                resource: buffer.0.as_entire_binding(),
            });
            buffer_layout.push(wgpu::BindGroupLayoutEntry {
                binding: buffer.1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        for buffer in self.output_buffers.iter() {
            buffer_list.push(wgpu::BindGroupEntry {
                binding: buffer.1,
                resource: buffer.0.as_entire_binding(),
            });
            buffer_layout.push(wgpu::BindGroupLayoutEntry {
                binding: buffer.1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&("Bind group layout for shader ".to_owned() + &self.name)),
            entries: &buffer_layout
        });

        self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&("Bind group for shader ".to_owned() + &self.name)),
            layout: &bind_group_layout,
            entries: &buffer_list
        }));

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        for entry in entry_points {
            self.pipeline.push((
                entry.to_string(),
                self.device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &self.shader,
                    entry_point: Some(entry),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
            })));
        }

        
    }

    pub fn execute(&self, name: &str, work_size: [u32; 3]) {
        let pipeline = &self.pipeline
            .iter()
            .find(|(entry_point, _pip)| entry_point == name)
            .expect("No pipeline for this entry point").1;

        let bind_group = self.bind_group
            .as_ref()
            .expect("No bind group");

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);

            // One invocation per element
            compute_pass.dispatch_workgroups(work_size[0], work_size[1], work_size[2]);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn retrieve_data<T: bytemuck::Pod>(&self, binding: u32, target: &mut [T]) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        let index = self.output_buffers.iter().enumerate()
            .find(|b| b.1.1 == binding)
            .expect("Cannot retrieve buffer - nothing binded").0;

        let gpu_buffer = &self.output_buffers[index].0;
        let staging_buffer = &self.staging_buffers[index];
        let buffer_size = &self.output_buffers_sizes[index];

        encoder.copy_buffer_to_buffer(
            gpu_buffer,
            0,
            staging_buffer,
            0,
            *buffer_size,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);

        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None
        });

        let data = buffer_slice.get_mapped_range();

        target.copy_from_slice(bytemuck::cast_slice(&data));

        drop(data);
        staging_buffer.unmap();
    }


}