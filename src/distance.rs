pub fn euclidean_3d_elementwise(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) -> f64 {
    (((x2 - x1).powi(2)) + ((y2 - y1).powi(2)) + ((z2 - z1).powi(2))).sqrt() 
}


pub fn euclidean_2d_elementwise(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    (((x2 - x1).powi(2)) + ((y2 - y1).powi(2))).sqrt() 
}
