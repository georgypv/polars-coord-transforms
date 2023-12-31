extern crate s2;
use s2::cellid::CellID;
use s2::latlng::LatLng;


pub fn lonlat_to_cellid_elementwise(lng: f64, lat: f64, level: u64) -> u64 {
    let cell_id = CellID::from(LatLng::from_degrees(lat, lng));
    cell_id.parent(level).0
}


pub fn cellid_to_lonlat_elementwise(cellid: u64) -> (f64, f64) {
    let lnglat = LatLng::from(CellID(cellid));
    let lat = lnglat.lat.deg();
    let lng = lnglat.lng.deg();
    (lng, lat)
}



#[cfg(test)]
mod s2_tests {

    use crate::s2_functions::{lonlat_to_cellid_elementwise, cellid_to_lonlat_elementwise};

    #[test]
    fn test_lonlat_to_cellid() {
        let lon: f64 = 36.077147686805766;
        let lat: f64 = 56.783927007002866;
        let level: u64 = 30;
        let expected_cellid: u64 = 5095400969591719543;

        assert_eq!(lonlat_to_cellid_elementwise(lon, lat, level), expected_cellid);
    }

    #[test]
    fn test_cellid_to_lonlat() {
        let cellid: u64 = 5095400969591719540;
        let expected_lonlat: (f64, f64) = (36.077147799420544, 56.78392700893653);
        
        assert_eq!(cellid_to_lonlat_elementwise(cellid), expected_lonlat)
    }

    #[test]
    fn s2_circular_transformation() {
        let cellid: u64 = 5095400969591719543; //level 30 S2 cell
        let (lon, lat) = cellid_to_lonlat_elementwise(cellid);

        assert_eq!(lonlat_to_cellid_elementwise(lon, lat, 30), cellid)
    }
    
}