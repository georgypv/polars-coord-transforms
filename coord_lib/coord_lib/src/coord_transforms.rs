extern crate nalgebra as na;

use na::{Vector3, Vector4, Quaternion, UnitQuaternion, Rotation3};
use map_3d::{ecef2geodetic, rad2deg, Ellipsoid};

pub fn enu_to_ecef_elementwise(enu_coords: Vec<f64>, rotation: Vec<f64>, offset: Vec<f64>) -> (f64, f64, f64) {

    let quat = UnitQuaternion::from_quaternion(Quaternion::from_vector(Vector4::from_vec(rotation)));
    let r = Rotation3::from(quat);
    let ecef_vector = r.transform_vector(&Vector3::from_vec(enu_coords)) + Vector3::from_vec(offset);

    (ecef_vector.x, ecef_vector.y, ecef_vector.z)
}

pub fn ecef_to_lla_elementwise(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let lla = ecef2geodetic(x, y, z, Ellipsoid::WGS84);
    (rad2deg(lla.1), rad2deg(lla.0), lla.2)
}

