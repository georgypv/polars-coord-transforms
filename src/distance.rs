#[derive(Debug, Clone, Copy)]
pub struct PointCoords {
    pub x: f64,
    pub y: f64,
}

pub fn euclidean_3d_elementwise(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) -> f64 {
    (((x2 - x1).powi(2)) + ((y2 - y1).powi(2)) + ((z2 - z1).powi(2))).sqrt()
}

pub fn euclidean_2d_elementwise(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    (((x2 - x1).powi(2)) + ((y2 - y1).powi(2))).sqrt()
}

pub fn cosine_similarity_2d_elementwise(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dot_product = (x1 * x2) + (y1 * y2);
    let magnitude1 = (x1.powi(2) + y1.powi(2)).powf(0.5);
    let magnitude2 = (x2.powi(2) + y2.powi(2)).powf(0.5);

    let res = if magnitude1 == 0.0 || magnitude2 == 0.0 {
        0.0
    } else {
        dot_product / (magnitude1 * magnitude2)
    };
    res
}

pub fn cosine_similarity_3d_elementwise(
    x1: f64,
    y1: f64,
    z1: f64,
    x2: f64,
    y2: f64,
    z2: f64,
) -> f64 {
    let dot_product = (x1 * x2) + (y1 * y2) + (z1 * z2);
    let magnitude1 = (x1.powi(2) + y1.powi(2) + z1.powi(2)).powf(0.5);
    let magnitude2 = (x2.powi(2) + y2.powi(2) + z2.powi(2)).powf(0.5);

    let res = if magnitude1 == 0.0 || magnitude2 == 0.0 {
        0.0
    } else {
        dot_product / (magnitude1 * magnitude2)
    };
    res
}

fn round(num: f64, precision: u8) -> f64 {
    let multiplier = 10f64.powi(precision as i32);
    (num * multiplier).round() / multiplier
}

pub fn dist_to_segment(point: PointCoords, start: PointCoords, end: PointCoords) -> f64 {
    let l2 = euclidean_2d_elementwise(start.x, start.y, end.x, end.y).powi(2);
    if l2 == 0.0 {
        return euclidean_2d_elementwise(point.x, point.y,  start.x, start.y).powi(2);
    }
    let t = ((point.x - start.x) * (end.x - start.x) + (point.y - start.y) * (end.y - start.y)) / l2;
    let t = t.max(0.0).min(1.0);
    let projection = PointCoords{x: start.x + t * (end.x - start.x), y: start.y + t * (end.y - start.y)};
    let dist_to_segment = euclidean_2d_elementwise(point.x, point.y, projection.x, projection.y);
    dist_to_segment
}

pub fn bboxes_2d_elementwise(box1: [PointCoords; 4], box2: [PointCoords; 4]) -> f64 {
    let mut min_distance = f64::MAX;

    // Sides of the first box
    for i in 0..4 {
        let start = box1[i];
        let end = box1[(i + 1) % 4];

        // Distance from each apex of the second box to the side of the first box
        for &point in &box2 {
            let distance = dist_to_segment(point, start, end);
            min_distance = min_distance.min(distance);
        }
    }

    // Sides of the second box
    for i in 0..4 {
        let start = box2[i];
        let end = box2[(i + 1) % 4];

        // Distance from each apex of the first box to the side of the second box
        for &point in &box1 {
            let distance = dist_to_segment(point, start, end);
            min_distance = min_distance.min(distance);
        }
    }

    round(min_distance, 5)
}


#[cfg(test)]
mod distance_tests {
    use crate::distance::{dist_to_segment, bboxes_2d_elementwise, PointCoords};

    #[test]
    fn test_dist_to_segment() {

        let point  =  PointCoords { x: 5.0, y: 2.0};
        let start  = PointCoords { x: 1.0, y: 0.0};
        let end = PointCoords { x: 3.0, y: 0.0};

        let expected_distance = 2.8284271247461903;

        assert_eq!(
            dist_to_segment(point, start, end),
                    expected_distance
                )
    }

    #[test]
    fn test_bboxes_2d_elementwise() {

        let bbox1 = [
            PointCoords {x: 2.0, y: 0.0,},
            PointCoords {x: 0.0, y: 3.0,},
            PointCoords {x: 2.0, y: 4.0,},
            PointCoords {x: 4.0, y: 1.0,},
        ];

        let bbox2 = [
            PointCoords {x: 3.0, y: 5.0,},
            PointCoords {x: 3.0, y: 8.0,},
            PointCoords {x: 5.0, y: 8.0,},
            PointCoords {x: 5.0, y: 5.0,},
        ];

        let expected_distance = 1.41421;

        assert_eq!(
            bboxes_2d_elementwise(bbox1, bbox2),
                    expected_distance
                )
    }
}
