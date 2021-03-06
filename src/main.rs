mod random;
mod scalar;
mod scene;

use clap::Parser;
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

const RR_STOP_PROBABILITY: f64 = 0.1;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, value_parser, default_value_t = 600)]
    width: u32,

    #[clap(short, long, value_parser, default_value_t = 600)]
    height: u32,

    #[clap(short, long, value_parser, default_value_t = 256)]
    samples: u32,
}

enum UserEvent {
    Frame(Vec<Vector3<f64>>),
}

struct Renderer {
    viewport: Viewport,
    camera: Camera,
    scene: Scene,
}

fn main() -> Result<(), Error> {
    dotenv::dotenv().ok();
    env_logger::init();
    let args = Args::parse();
    let event_loop = EventLoop::with_user_event();
    let event_loop_proxy = event_loop.create_proxy();
    let window = {
        let size = LogicalSize::new(args.width as f64, args.height as f64);
        WindowBuilder::new()
            .with_title("Raytracing demo")
            .with_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(args.width, args.height, surface_texture)?
    };
    let renderer = Renderer::new(Viewport::new(args.width as usize, args.height as usize));
    std::thread::spawn(move || {
        let mut frame = Vec::new();
        frame.resize_with((args.width * args.height) as usize, || Vector3::zeros());

        let inv_samples = 1.0 / args.samples as f64;
        for _ in 0..args.samples {
            renderer.draw(&mut frame, inv_samples);
            event_loop_proxy
                .send_event(UserEvent::Frame(frame.clone()))
                .ok();
        }
    });

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            winit::event::WindowEvent::Resized(new_size) => {
                pixels.resize_surface(new_size.width, new_size.height);
            }
            winit::event::WindowEvent::CloseRequested => {
                *control_flow = ControlFlow::Exit;
            }
            _ => {}
        },
        Event::RedrawRequested(_) => {
            if pixels
                .render()
                .map_err(|e| error!("pixels.render() failed: {}", e))
                .is_err()
            {
                *control_flow = ControlFlow::Exit;
                return;
            }
        }
        Event::UserEvent(user_event) => match user_event {
            UserEvent::Frame(frame) => {
                let pixels_frame = pixels.get_frame();
                for (src, dst) in frame.into_iter().zip(pixels_frame.chunks_exact_mut(4)) {
                    dst.copy_from_slice(&[
                        src.x.min(255.0) as u8,
                        src.y.min(255.0) as u8,
                        src.z.min(255.0) as u8,
                        0xff,
                    ]);
                }
                window.request_redraw();
                log::info!("frame received");
            }
        },
        _ => {}
    });
}

impl Renderer {
    /// Create a new `World` instance that can draw a moving box.
    fn new(viewport: Viewport) -> Self {
        let mut scene = Scene::new();
        // Middle sphere
        scene.add(
            Sphere::new(1.05, Vector3::new(-0.75, -1.45, -4.4)),
            Material::new(0.0, MaterialKind::Specular),
        );
        // Right sphere
        scene.add(
            Sphere::new(0.5, Vector3::new(2.0, -2.05, -3.7)),
            Material::new(
                0.0,
                MaterialKind::Refractive {
                    refraction_index: 1.5,
                },
            ),
        );
        // Left sphere
        scene.add(
            Sphere::new(0.6, Vector3::new(-1.75, -1.95, -3.1)),
            Material::new(
                0.0,
                MaterialKind::Diffuse {
                    color: Vector3::new(4.0, 4.0, 12.0),
                },
            ),
        );
        // Light
        scene.add(
            Sphere::new(0.5, Vector3::new(0.0, 1.9, -3.0)),
            Material::new(
                10000.0,
                MaterialKind::Diffuse {
                    color: Vector3::zeros(),
                },
            ),
        );

        // Bottom plane
        scene.add(
            Plane::new(2.5, Vector3::new(0.0, 1.0, 0.0)),
            Material::new(
                0.0,
                MaterialKind::Diffuse {
                    color: Vector3::new(6.0, 6.0, 6.0),
                },
            ),
        );
        // Back plane
        scene.add(
            Plane::new(5.5, Vector3::new(0.0, 0.0, 1.0)),
            Material::new(
                0.0,
                MaterialKind::Diffuse {
                    color: Vector3::new(6.0, 6.0, 6.0),
                },
            ),
        );
        // Left plane
        scene.add(
            Plane::new(2.75, Vector3::new(1.0, 0.0, 0.0)),
            Material::new(
                0.0,
                MaterialKind::Diffuse {
                    color: Vector3::new(10.0, 2.0, 2.0),
                },
            ),
        );
        // Right plane
        scene.add(
            Plane::new(2.75, Vector3::new(-1.0, 0.0, 0.0)),
            Material::new(
                0.0,
                MaterialKind::Diffuse {
                    color: Vector3::new(2.0, 10.0, 2.0),
                },
            ),
        );
        // Ceiling plane
        scene.add(
            Plane::new(3.0, Vector3::new(0.0, -1.0, 0.0)),
            Material::new(
                0.0,
                MaterialKind::Diffuse {
                    color: Vector3::new(6.0, 6.0, 6.0),
                },
            ),
        );
        // Front plane
        scene.add(
            Plane::new(0.5, Vector3::new(0.0, 0.0, -1.0)),
            Material::new(
                0.0,
                MaterialKind::Diffuse {
                    color: Vector3::new(6.0, 6.0, 6.0),
                },
            ),
        );

        let camera = Camera::new(std::f64::consts::FRAC_PI_4);
        Self {
            scene,
            camera,
            viewport,
        }
    }

    fn draw(&self, frame: &mut Vec<Vector3<f64>>, weight: f64) {
        let viewport_width = self.viewport.width;
        frame.par_iter_mut().enumerate().for_each_init(
            || Random::new(),
            |rng, (i, pixel)| {
                let x = (i % viewport_width) as f64;
                let y = (i / viewport_width) as f64;

                let mut ray_direction = self.camera.ray(&self.viewport, x, y);
                ray_direction.x += (rng.sample() * 2.0 - 1.0) / 700.0;
                ray_direction.y += (rng.sample() * 2.0 - 1.0) / 700.0;
                let ray = Ray::new(Vector3::zeros(), ray_direction);

                *pixel += self.trace(&ray, rng, 0) * weight;
            },
        );
    }

    fn trace(&self, ray: &Ray, rng: &mut Random, depth: usize) -> Vector3<f64> {
        let mut rr_factor = 1.0;
        if depth >= 5 {
            if rng.sample() <= RR_STOP_PROBABILITY {
                return Vector3::zeros();
            }
            rr_factor = 1.0 / (1.0 - RR_STOP_PROBABILITY);
        }

        match self.scene.trace(ray) {
            Some(trace_result) => {
                let material = trace_result.material;
                let mut result = Vector3::zeros();
                result += Vector3::from_element(material.emission) * rr_factor;

                let scattered_ray = material.scatter(
                    ray,
                    &trace_result.hit_point,
                    &trace_result.normal,
                    trace_result.dot_product,
                    rng,
                );
                let ambient = self.trace(&scattered_ray, rng, depth + 1);
                result +=
                    material.get_color(&ambient, &scattered_ray, &trace_result.normal, rr_factor);
                result
            }
            None => Vector3::zeros(),
        }
    }
}
