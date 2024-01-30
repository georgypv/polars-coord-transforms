pub fn euclidean_3d_elementwise(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) -> f64 {
    (((x2 - x1).powi(2)) + ((y2 - y1).powi(2)) + ((z2 - z1).powi(2))).sqrt() 
}


pub fn euclidean_2d_elementwise(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    (((x2 - x1).powi(2)) + ((y2 - y1).powi(2))).sqrt() 
}


pub fn cosine_similarity_2d_elementwise(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dot_product = (x1*x2) + (y1*y2);
    let magnitude1 = (x1.powi(2) + y1.powi(2)).powf(0.5);
    let magnitude2 = (x2.powi(2) + y2.powi(2)).powf(0.5);

    let res = if magnitude1 == 0.0 || magnitude2 == 0.0 {
        0.0
    } else {
        dot_product / (magnitude1*magnitude2)
    };
    res
}

pub fn cosine_similarity_3d_elementwise(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) -> f64 {
    let dot_product = (x1*x2) + (y1*y2) + (z1*z2);
    let magnitude1 = (x1.powi(2) + y1.powi(2) + z1.powi(2)).powf(0.5);
    let magnitude2 = (x2.powi(2) + y2.powi(2) + z2.powi(2)).powf(0.5);

    let res = if magnitude1 == 0.0 || magnitude2 == 0.0 {
        0.0
    } else {
        dot_product / (magnitude1*magnitude2)
    };
    res
}