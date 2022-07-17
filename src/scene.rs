use nalgebra::Vector3;

use crate::{random::Random, scalar::EPSILON};

#[derive(Debug)]
pub struct Ray {
    pub origin: Vector3<f64>,
    pub direction: Vector3<f64>,
}

impl Ray {
    pub fn new(origin: Vector3<f64>, direction: Vector3<f64>) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MaterialKind {
    Diffuse,
    Specular,
    Refractive { refraction_index: f64 },
}

#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub color: Vector3<f64>,
    pub emission: f64,
    pub kind: MaterialKind,
}

impl Material {
    pub fn new(color: Vector3<f64>, emission: f64, kind: MaterialKind) -> Self {
        Self {
            color,
            emission,
            kind,
        }
    }

    pub fn hit(
        &self,
        ray: &Ray,
        hit_point: &Vector3<f64>,
        normal: &Vector3<f64>,
        dot_product: f64,
        rng: &mut Random,
    ) -> Ray {
        match self.kind {
            MaterialKind::Diffuse => {
                let (rot_x, rot_y) = make_orthonormal_system(&normal);
                let sampled_dir = hemisphere(rng.sample(), rng.sample());
                let rotated_dir = Vector3::new(
                    Vector3::new(rot_x.x, rot_y.x, normal.x).dot(&sampled_dir),
                    Vector3::new(rot_x.y, rot_y.y, normal.y).dot(&sampled_dir),
                    Vector3::new(rot_x.z, rot_y.z, normal.z).dot(&sampled_dir),
                );
                Ray::new(hit_point.clone(), rotated_dir)
            }
            MaterialKind::Specular => Ray::new(
                hit_point.clone(),
                ray.direction - 2.0 * dot_product * normal,
            ),
            MaterialKind::Refractive {
                refraction_index: mut n,
            } => {
                let r0 = (1.0 - n) / (1.0 + n);
                let r0 = r0 * r0;
                let mut normal = normal.clone();
                // inside the medium
                if normal.dot(&ray.direction) > 0.0 {
                    normal = -normal;
                    n = 1.0 / n;
                }
                n = 1.0 / n;
                let cost1 = -dot_product;
                let cost2 = 1.0 - n * n * (1.0 - cost1 * cost1);
                let rprob = r0 + (1.0 - r0) * (1.0 - cost1).powi(5);
                let ray_direction = if cost2 > 0.0 && rng.sample() > rprob {
                    // refraction
                    ray.direction * n + normal * (n * cost1 - cost2.sqrt())
                } else {
                    //reflection
                    ray.direction + normal * (cost1 * 2.0)
                };
                Ray::new(hit_point.clone(), ray_direction)
            }
        }
    }
}

pub trait Geometry {
    fn intersect(&self, ray: &Ray) -> Option<f64>;
    fn normal(&self, p: Vector3<f64>) -> Vector3<f64>;
}

#[derive(Debug)]
pub struct Plane {
    normal: Vector3<f64>,
    d: f64,
}

impl Plane {
    pub fn new(d: f64, normal: Vector3<f64>) -> Self {
        Self { normal, d }
    }
}

impl Geometry for Plane {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        let d0 = self.normal.dot(&ray.direction);
        if d0.abs() > EPSILON {
            let t = -(self.normal.dot(&ray.origin) + self.d) / d0;
            if t > EPSILON {
                Some(t)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn normal(&self, _: Vector3<f64>) -> Vector3<f64> {
        self.normal
    }
}

#[derive(Debug)]
pub struct Sphere {
    center: Vector3<f64>,
    radius: f64,
}

impl Sphere {
    pub fn new(radius: f64, center: Vector3<f64>) -> Self {
        Self { center, radius }
    }
}

impl Geometry for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        let oc = ray.origin - self.center;
        let a = 1.0;
        let half_b = oc.dot(&ray.direction);
        let c = oc.magnitude_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;
        match discriminant {
            x if x < 0.0 => None,
            x => {
                let sqrtd = x.sqrt();
                Some((-half_b - sqrtd).min(-half_b + sqrtd) / a)
            }
        }
    }

    fn normal(&self, p: Vector3<f64>) -> Vector3<f64> {
        (p - self.center).normalize()
    }
}

pub enum Object {
    Plane(Plane),
    Sphere(Sphere),
}

impl From<Sphere> for Object {
    fn from(sphere: Sphere) -> Self {
        Self::Sphere(sphere)
    }
}

impl From<Plane> for Object {
    fn from(plane: Plane) -> Self {
        Self::Plane(plane)
    }
}

impl Geometry for Object {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        match self {
            Object::Plane(plane) => plane.intersect(ray),
            Object::Sphere(sphere) => sphere.intersect(ray),
        }
    }

    fn normal(&self, p: Vector3<f64>) -> Vector3<f64> {
        match self {
            Object::Plane(plane) => plane.normal(p),
            Object::Sphere(sphere) => sphere.normal(p),
        }
    }
}

pub type ObjectWithMaterial = (Object, Material);

pub struct Viewport {
    pub width: usize,
    pub height: usize,
}

impl Viewport {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
}

pub struct Camera {
    fovx: f64,
}

impl Camera {
    pub fn new(fovx: f64) -> Self {
        Self { fovx }
    }

    pub fn ray(&self, viewport: &Viewport, x: f64, y: f64) -> Vector3<f64> {
        let w = viewport.width as f64;
        let h = viewport.height as f64;
        let fovx = self.fovx;
        let fovy = h * fovx / w;
        Vector3::new(
            (2.0 * x - w) * fovx.tan() / w,
            -(2.0 * y - h) * fovy.tan() / h,
            -1.0,
        )
    }
}

pub struct TraceResult<'a> {
    pub distance: f64,
    pub hit_point: Vector3<f64>,
    pub normal: Vector3<f64>,
    pub material: &'a Material,
    pub dot_product: f64,
}

impl<'a> TraceResult<'a> {
    pub fn new(
        distance: f64,
        hit_point: Vector3<f64>,
        normal: Vector3<f64>,
        dot_product: f64,
        material: &'a Material,
    ) -> Self {
        Self {
            distance,
            material,
            hit_point,
            normal,
            dot_product,
        }
    }
}

pub struct Scene {
    objects: Vec<ObjectWithMaterial>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    pub fn add<T: Into<Object>>(&mut self, object: T, material: Material) {
        self.objects.push((object.into(), material));
    }

    pub fn trace(&self, ray: &Ray) -> Option<TraceResult> {
        self.objects
            .iter()
            .filter_map(|(object, material)| match object.intersect(ray) {
                Some(distance) => {
                    let hit_point = ray.origin + ray.direction * distance;
                    let normal = object.normal(hit_point);
                    Some(TraceResult::new(
                        distance,
                        hit_point,
                        normal,
                        normal.dot(&ray.direction),
                        material,
                    ))
                }
                None => None,
            })
            .min_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

fn hemisphere(u1: f64, u2: f64) -> Vector3<f64> {
    let r = (1.0 - u1 * u1).sqrt();
    let phi = 2.0 * std::f64::consts::PI * u2;
    Vector3::new(phi.cos() * r, phi.sin() * r, u1)
}

fn make_orthonormal_system(v: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let v2 = if v.x.abs() > v.y.abs() {
        let inv_len = 1.0 / v.xy().magnitude();
        Vector3::new(-v.z * inv_len, 0.0, v.x * inv_len)
    } else {
        let inv_len = 1.0 / v.yz().magnitude();
        Vector3::new(0.0, v.z * inv_len, -v.y * inv_len)
    };
    let v3 = v.cross(&v2);
    (v2, v3)
}
