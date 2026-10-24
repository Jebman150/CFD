use std::sync::Arc;

use tracing::{Level, event, instrument};

pub struct GPUContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>
}

impl GPUContext {
    #[instrument]
    pub async fn new() -> anyhow::Result<Self> {
        event!(Level::INFO, "Creating GPU instance");
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await?;

        event!(Level::INFO, "Found GPU adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::POLYGON_MODE_LINE,
                ..Default::default()
            })
            .await?;

        event!(Level::INFO, "Found GPU device");

        Ok(Self {
            instance: instance,
            adapter: adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }
}
