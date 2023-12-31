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
