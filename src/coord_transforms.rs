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


#[cfg(test)]
mod transform_tests { 
    use crate::coord_transforms::{enu_to_ecef_elementwise, ecef_to_lla_elementwise};

    #[test]
    fn test_enu_to_ecef() {
        let enu_coords: Vec<f64> = vec![-97066.730132, 122807.787398,	-1888.737721];
        let rotation: Vec<f64> = vec![0.13007119, 0.26472049, 0.85758219, 0.42137553];
        let offset: Vec<f64> = vec![2852423.40536658, 2201848.41975346, 5245234.74365368];

        let expected_result: (f64, f64, f64) = (2830593.6327610738, 2062375.5703225536, 5312896.0721501345);

        assert_eq!(enu_to_ecef_elementwise(enu_coords, rotation, offset), expected_result)

    }

    #[test]
    fn test_ecef_to_lla() {
        
        let ecef_coords: (f64, f64, f64) = (2830593.6327610738, 2062375.5703225536, 5312896.0721501345);
        let expected_result: (f64, f64, f64) = (36.077147686805766, 56.783927007002866, 165.8986865637805);

        assert_eq!(ecef_to_lla_elementwise(ecef_coords.0, ecef_coords.1, ecef_coords.2), expected_result)
    }

}
