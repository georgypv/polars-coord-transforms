extern crate nalgebra as na;

use utm::{to_utm_wgs84_no_zone, lat_lon_to_zone_number};
use na::{Vector3, Vector4, Quaternion, UnitQuaternion, Rotation3};
use map_3d::{ecef2geodetic, geodetic2ecef, rad2deg, deg2rad, Ellipsoid};

fn rotation_from_quat(rotation: Vec<f64>) -> na::Rotation<f64, 3> {
    let quat = UnitQuaternion::from_quaternion(Quaternion::from_vector(Vector4::from_vec(rotation)));
    Rotation3::from(quat)
} 

pub fn map_to_ecef_elementwise(map_coords: Vec<f64>, rotation: Vec<f64>, offset: Vec<f64>) -> (f64, f64, f64) {

    let r: na::Rotation<f64, 3> = rotation_from_quat(rotation);
    let ecef_vector = r.transform_vector(&Vector3::from_vec(map_coords)) + Vector3::from_vec(offset);

    (ecef_vector.x, ecef_vector.y, ecef_vector.z)
}

pub fn ecef_to_map_elementwise(ecef_coords: Vec<f64>, rotation: Vec<f64>, offset: Vec<f64>) -> (f64, f64, f64) {
    let r_inverse: na::Rotation<f64, 3> = rotation_from_quat(rotation).inverse();
    
    let map_vector = r_inverse.transform_vector(&(Vector3::from_vec(ecef_coords) - Vector3::from_vec(offset)));
    (map_vector.x, map_vector.y, map_vector.z)

}

pub fn ecef_to_lla_elementwise(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let lla = ecef2geodetic(x, y, z, Ellipsoid::WGS84);
    (rad2deg(lla.1), rad2deg(lla.0), lla.2)
}

pub fn lla_to_ecef_elementwise(lon: f64, lat: f64, alt: f64) -> (f64, f64, f64) {
    let (x, y, z) = geodetic2ecef(deg2rad(lat), deg2rad(lon), alt, Ellipsoid::WGS84);
    (x, y, z)
}



pub fn lla_to_utm_zone_number_elementwise(lon: f64, lat: f64) -> u8 {
    let zone_number = lat_lon_to_zone_number(lat, lon);
    zone_number
}   


pub fn lla_to_utm_elementwise(lon: f64, lat: f64, alt: f64) -> (f64, f64, f64) {
    let (northing, easting, _meridian_convergence) = to_utm_wgs84_no_zone(lat, lon);
    (easting, northing, alt)
}


pub fn rotate_map_coords_elementwise(map_coords: Vec<f64>, rotation: Vec<f64>, scale: Vec<f64>) -> (f64, f64, f64) {
    let r: na::Rotation<f64, 3> = rotation_from_quat(rotation);
    let scale_rotated = r.transform_vector(&Vector3::from_vec(scale));
    let map_coords_rotated = Vector3::from_vec(map_coords) + scale_rotated;
    (map_coords_rotated.x, map_coords_rotated.y, map_coords_rotated.z)

}


#[cfg(test)]
mod transform_tests { 
    use crate::coord_transforms::{map_to_ecef_elementwise, ecef_to_lla_elementwise};

    #[test]
    fn test_map_to_ecef() {
        let map_coords: Vec<f64> = vec![-97066.730132, 122807.787398,	-1888.737721];
        let rotation: Vec<f64> = vec![0.13007119, 0.26472049, 0.85758219, 0.42137553];
        let offset: Vec<f64> = vec![2852423.40536658, 2201848.41975346, 5245234.74365368];

        let expected_result: (f64, f64, f64) = (2830593.6327610738, 2062375.5703225536, 5312896.0721501345);

        assert_eq!(map_to_ecef_elementwise(map_coords, rotation, offset), expected_result)

    }

    #[test]
    fn test_ecef_to_lla() {
        
        let ecef_coords: (f64, f64, f64) = (2830593.6327610738, 2062375.5703225536, 5312896.0721501345);
        let expected_result: (f64, f64, f64) = (36.077147686805766, 56.783927007002866, 165.8986865637805);

        assert_eq!(ecef_to_lla_elementwise(ecef_coords.0, ecef_coords.1, ecef_coords.2), expected_result)
    }

}


