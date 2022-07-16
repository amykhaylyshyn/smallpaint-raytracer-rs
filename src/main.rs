mod random;
mod scalar;
mod scene;

use log::error;
use nalgebra::Vector3;
use pixels::{Error, Pixels, SurfaceTexture};
use random::Random;
use rayon::prelude::*;
use scene::Scene;
use scene::*;
use winit::dpi::LogicalSize;
use winit::event::Event;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const SAMPLES: u32 = 2;
const RR_STOP_PROBABILITY: f64 = 0.1;

/// Representation of the application state. In this example, a box will bounce around the screen.
struct World {
    viewport: Viewport,
    camera: Camera,
    scene: Scene,
}

fn main() -> Result<(), Error> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Raytracing demo")
            .with_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture)?
    };
    let mut world = World::new(Viewport::new(WIDTH as f64, HEIGHT as f64));
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            winit::event::WindowEvent::Resized(new_size) => {
                pixels.resize_surface(new_size.width, new_size.height);
                world.resize_viewport(new_size.width as f64, new_size.height as f64);
            }
            winit::event::WindowEvent::CloseRequested => {
                *control_flow = ControlFlow::Exit;
            }
            _ => {}
        },
        Event::RedrawRequested(_) => {
            world.draw(pixels.get_frame());
            if pixels
                .render()
                .map_err(|e| error!("pixels.render() failed: {}", e))
                .is_err()
            {
                *control_flow = ControlFlow::Exit;
                return;
            }
        }
        Event::MainEventsCleared => {
            // window.request_redraw();
        }
        _ => {}
    });
}

impl World {
    /// Create a new `World` instance that can draw a moving box.
    fn new(viewport: Viewport) -> Self {
        let mut scene = Scene::new();
        // Middle sphere
        scene.add(
            Sphere::new(1.05, Vector3::new(-0.75, -1.45, -4.4)),
            Material::new(Vector3::new(4.0, 8.0, 4.0), 0.0, MaterialKind::Specular),
        );
        // Right sphere
        scene.add(
            Sphere::new(0.5, Vector3::new(2.0, -2.05, -3.7)),
            Material::new(
                Vector3::new(10.0, 10.0, 1.0),
                0.0,
                MaterialKind::Refractive {
                    refraction_index: 1.5,
                },
            ),
        );
        // Left sphere
        scene.add(
            Sphere::new(0.6, Vector3::new(0.0, -1.95, -3.1)),
            Material::new(Vector3::new(4.0, 4.0, 12.0), 0.0, MaterialKind::Diffuse),
        );
        // Light
        scene.add(
            Sphere::new(0.5, Vector3::new(0.0, 1.9, -3.0)),
            Material::new(Vector3::zeros(), 10000.0, MaterialKind::Diffuse),
        );

        // Bottom plane
        scene.add(
            Plane::new(2.5, Vector3::new(0.0, 1.0, 0.0)),
            Material::new(Vector3::new(6.0, 6.0, 6.0), 0.0, MaterialKind::Diffuse),
        );
        // Back plane
        scene.add(
            Plane::new(5.5, Vector3::new(0.0, 0.0, 1.0)),
            Material::new(Vector3::new(6.0, 6.0, 6.0), 0.0, MaterialKind::Diffuse),
        );
        // Left plane
        scene.add(
            Plane::new(2.75, Vector3::new(1.0, 0.0, 0.0)),
            Material::new(Vector3::new(10.0, 2.0, 2.0), 0.0, MaterialKind::Diffuse),
        );
        // Right plane
        scene.add(
            Plane::new(2.75, Vector3::new(-1.0, 0.0, 0.0)),
            Material::new(Vector3::new(2.0, 10.0, 2.0), 0.0, MaterialKind::Diffuse),
        );
        // Ceiling plane
        scene.add(
            Plane::new(3.0, Vector3::new(0.0, -1.0, 0.0)),
            Material::new(Vector3::new(6.0, 6.0, 6.0), 0.0, MaterialKind::Diffuse),
        );
        // Front plane
        scene.add(
            Plane::new(0.5, Vector3::new(0.0, 0.0, -1.0)),
            Material::new(Vector3::new(6.0, 6.0, 6.0), 0.0, MaterialKind::Diffuse),
        );

        let camera = Camera::new(std::f64::consts::FRAC_PI_3);
        Self {
            scene,
            camera,
            viewport,
        }
    }

    fn resize_viewport(&mut self, width: f64, height: f64) {
        self.viewport = Viewport::new(width, height);
    }

    /// Draw the `World` state to the frame buffer.
    ///
    /// Assumes the default texture format: `wgpu::TextureFormat::Rgba8UnormSrgb`
    fn draw(&self, frame: &mut [u8]) {
        let viewport_width = self.viewport.width as usize;
        let viewport_height = self.viewport.height as usize;
        let inv_samples = 1.0 / SAMPLES as f64;
        frame.par_chunks_exact_mut(4).enumerate().for_each_init(
            || Random::new(),
            |rng, (i, pixel)| {
                let x = (i % viewport_height) as f64;
                let y = (i / viewport_width) as f64;

                let mut color = Vector3::zeros();
                for _ in 0..SAMPLES {
                    let mut ray_direction = self.camera.ray(&self.viewport, x, y);
                    ray_direction.x += (rng.sample() * 2.0 - 1.0) / 700.0;
                    ray_direction.y += (rng.sample() * 2.0 - 1.0) / 700.0;
                    let ray = Ray::new(Vector3::zeros(), ray_direction);

                    color += self.trace(&ray, rng, Vector3::zeros(), 0) * inv_samples;
                }

                let rgba = [
                    color.x.min(255.0) as u8,
                    color.y.min(255.0) as u8,
                    color.z.min(255.0) as u8,
                    0xff,
                ];
                pixel.copy_from_slice(&rgba);
            },
        );
    }

    fn trace(
        &self,
        ray: &Ray,
        rng: &mut Random,
        color: Vector3<f64>,
        depth: usize,
    ) -> Vector3<f64> {
        let mut rr_factor = 1.0;
        if depth >= 5 {
            if rng.sample() > RR_STOP_PROBABILITY {
                return color;
            }
            rr_factor = 1.0 / (1.0 - RR_STOP_PROBABILITY);
        }

        match self.scene.trace(ray) {
            Some(trace_result) => {
                let material = trace_result.material;
                let mut result = color;
                result += Vector3::new(material.emission, material.emission, material.emission)
                    * rr_factor;

                let new_ray = material.hit(
                    ray,
                    &trace_result.hit_point,
                    &trace_result.normal,
                    trace_result.dot_product,
                    rng,
                );
                let color2 = self.trace(&new_ray, rng, result, depth + 1);
                match material.kind {
                    MaterialKind::Diffuse => {
                        result += Vector3::new(
                            color2.x * material.color.x,
                            color2.y * material.color.y,
                            color2.z * material.color.z,
                        ) * rr_factor
                            * 0.1
                            * new_ray.direction.dot(&trace_result.normal);
                    }
                    MaterialKind::Specular => {
                        result += color2 * rr_factor;
                    }
                    MaterialKind::Refractive { .. } => {
                        result += color2 * 1.15 * rr_factor;
                    }
                }

                result
            }
            None => color,
        }
    }
}
