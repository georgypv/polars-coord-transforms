use itertools::Itertools;
use polars::prelude::*;
use polars::datatypes::DataType;
use pyo3_polars::derive::polars_expr;

use itertools::izip;
use serde::Deserialize;

use crate::s2_functions::*;
use crate::coord_transforms::*;


// SSNameSpace
#[derive(Deserialize)]
struct S2Kwargs {
    level: u64
}


#[polars_expr(output_type=UInt64)]
fn lonlat_to_cellid(inputs: &[Series], kwargs: S2Kwargs) -> PolarsResult<Series> {

    let lonlat_ca = inputs[0].struct_()?;

    let lon = lonlat_ca.field_by_name("lon")?;
    let lat = lonlat_ca.field_by_name("lat")?;

    let lon: Series = match lon.dtype() {
        DataType::Float32 => lon.cast(&DataType::Float64)?,
        DataType::Float64 => lon,
        _ => polars_bail!(InvalidOperation:"lon must be float32 or float64!"),
    };

    let lat: Series = match lat.dtype() {
        DataType::Float32 => lat.cast(&DataType::Float64)?,
        DataType::Float64 => lat,
        _ => polars_bail!(InvalidOperation:"lat must be float32 or float64!"),
    };

    let lon_ca = lon.f64()?;
    let lat_ca = lat.f64()?;

    let iter = lon_ca.into_iter()
        .zip(lat_ca.into_iter())
        .map(|(lon_opt, lat_opt)| match (lon_opt, lat_opt) {
            (Some(lon), Some(lat)) => lonlat_to_cellid_elementwise(lon, lat, kwargs.level),
            _ => 0
        });
    let out_ca: ChunkedArray<UInt64Type> = iter.collect_ca_with_dtype("s2_cellid", DataType::UInt64);
    Ok(out_ca.into_series())
    
}


fn cellid_to_lonlat_output(_: &[Field]) -> PolarsResult<Field> {
    let v: Vec<Field> = vec![
        Field::new("lon", DataType::Float64),
        Field::new("lat", DataType::Float64),
    ];
    Ok(Field::new("coordinates", DataType::Struct(v)))
}

#[polars_expr(output_type_func=cellid_to_lonlat_output)]
fn cellid_to_lonlat(inputs: &[Series]) -> PolarsResult<Series> {

    let ca: &ChunkedArray<UInt64Type> = inputs[0].u64()?;

    let mut longitude: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("lon", ca.len());
    let mut latitude: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("lat", ca.len());

    
    for cellid_op in ca.into_iter() { 
            match cellid_op {
                Some(cellid) => {
                    let (lon, lat) = cellid_to_lonlat_elementwise(cellid);
                    longitude.append_value(lon);
                    latitude.append_value(lat)
                },
                _ => {
                    longitude.append_null();
                    latitude.append_null();
                }
            }
        }
    
    let ser_lon = longitude.finish().into_series();
    let ser_lat = latitude.finish().into_series();
    let out_chunked = StructChunked::new("coordinates", &[ser_lon, ser_lat])?;
    Ok(out_chunked.into_series())

}   

#[polars_expr(output_type=Boolean)]
fn cell_contains_point(inputs: &[Series]) -> PolarsResult<Series> {
    let cell_ca: &ChunkedArray<UInt64Type> = inputs[0].u64()?;

    let lonlat_ca = inputs[1].struct_()?;

    let lon_ser = lonlat_ca.field_by_name("lon")?;
    let lat_ser = lonlat_ca.field_by_name("lat")?;
    
    let lon_ca = lon_ser.f64()?;
    let lat_ca = lat_ser.f64()?;

    let out_ca: ChunkedArray<BooleanType> = izip!(cell_ca.into_iter(), lon_ca.into_iter(), lat_ca.into_iter()).map(
        | (cellid_op, lon_op, lat_op) | match (cellid_op, lon_op, lat_op) {
         (Some(cellid), Some(lon), Some(lat)) => cell_contains_point_elementwise(cellid, lon, lat),
         _ => false
        }   
    ).collect_ca(cell_ca.name());

    Ok(out_ca.into_series())
}

fn cellid_to_vertices_output(_: &[Field]) -> PolarsResult<Field> {
    let mut v: Vec<Field> = vec![];

    for i in 0..4 {
        v.push(
            Field::new(&format!("v{i}_lon"), DataType::Float64),
        );
        v.push(
            Field::new(&format!("v{i}_lat"), DataType::Float64),
        );

    }
    Ok(Field::new("vertices", DataType::Struct(v)))
}

#[polars_expr(output_type_func=cellid_to_vertices_output)]
fn cellid_to_vertices(inputs: &[Series]) -> PolarsResult<Series> {
    let cell_ca: &ChunkedArray<UInt64Type> = inputs[0].u64()?;

    let mut v_lon: Vec<PrimitiveChunkedBuilder<Float64Type>> = Vec::with_capacity(4);
    let mut v_lat: Vec<PrimitiveChunkedBuilder<Float64Type>> = Vec::with_capacity(4);

    for i in 0..4 {
        v_lon.push(
            PrimitiveChunkedBuilder::new(&format!("v{i}_lon"), cell_ca.len())
        );
        v_lat.push(
            PrimitiveChunkedBuilder::new(&format!("v{i}_lat"), cell_ca.len())
        );
    }    

    for cell_op in cell_ca.into_iter() {
        match cell_op {
            Some(cellid) => {
                let vertices: Vec<(f64, f64)> = cellid_to_vertices_elementwise(cellid);
                for i in 0..4 {
                    let (lon, lat) = vertices[i];
                    v_lon[i].append_value(lon);
                    v_lat[i].append_value(lat);
                    }
                },
            _ => {
                for i in 0..4 {
                        v_lon[i].append_null();
                        v_lat[i].append_null();
                    }
                }
        }
    }

    let v_coords_ser: Vec<Series> = v_lon.into_iter().zip(v_lat.into_iter()).flat_map(
        |(cb_lon, cb_lat)| [cb_lon.finish().into_series(), cb_lat.finish().into_series()].into_iter()
    ).collect_vec();

    let out_chunked = StructChunked::new("vertices", &v_coords_ser[..])?;
    Ok(out_chunked.into_series())
}


//TransfromNameSpace
fn ecef_output(_: &[Field]) -> PolarsResult<Field> {
    let v: Vec<Field> = vec![
        Field::new("x", DataType::Float64),
        Field::new("y", DataType::Float64),
        Field::new("z", DataType::Float64),
    ];
    Ok(Field::new("ecef", DataType::Struct(v)))
}

#[polars_expr(output_type_func=ecef_output)]
fn map_to_ecef(inputs: &[Series]) -> PolarsResult<Series> {

    let map_ca = inputs[0].struct_()?;
    let rotation_ca = inputs[1].struct_()?;
    let offset_ca = inputs[2].struct_()?;

    let map_x_ser = map_ca.field_by_name("x")?;
    let map_y_ser = map_ca.field_by_name("y")?;
    let map_z_ser = map_ca.field_by_name("z")?;
    let rotation_x_ser = rotation_ca.field_by_name("x")?;
    let rotation_y_ser = rotation_ca.field_by_name("y")?;
    let rotation_z_ser = rotation_ca.field_by_name("z")?;
    let rotation_w_ser = rotation_ca.field_by_name("w")?;
    let offset_x_ser = offset_ca.field_by_name("x")?;
    let offset_y_ser = offset_ca.field_by_name("y")?;
    let offset_z_ser = offset_ca.field_by_name("z")?;

    let map_x: &ChunkedArray<Float64Type>= map_x_ser.f64()?;
    let map_y: &ChunkedArray<Float64Type>= map_y_ser.f64()?;
    let map_z: &ChunkedArray<Float64Type>= map_z_ser.f64()?;
    let rotation_x: &ChunkedArray<Float64Type>= rotation_x_ser.f64()?;
    let rotation_y: &ChunkedArray<Float64Type>= rotation_y_ser.f64()?;
    let rotation_z: &ChunkedArray<Float64Type>= rotation_z_ser.f64()?;
    let rotation_w: &ChunkedArray<Float64Type>= rotation_w_ser.f64()?;
    let offset_x: &ChunkedArray<Float64Type>= offset_x_ser.f64()?;
    let offset_y: &ChunkedArray<Float64Type>= offset_y_ser.f64()?;
    let offset_z: &ChunkedArray<Float64Type>= offset_z_ser.f64()?;

    let mut ecef_x: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("x", map_ca.len());
    let mut ecef_y: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("y", map_ca.len());
    let mut ecef_z: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("z", map_ca.len());

    for (map_x_val, map_y_val, map_z_val, rotation_x_val, rotation_y_val, rotation_z_val, rotation_w_val, offset_x_val, offset_y_val, offset_z_val) 
        in izip!(map_x, map_y, map_z, rotation_x, rotation_y, rotation_z, rotation_w, offset_x, offset_y, offset_z) {
            let map_vec = vec![map_x_val.unwrap(), map_y_val.unwrap(), map_z_val.unwrap()];
            let rotation_vec = vec![rotation_x_val.unwrap(), rotation_y_val.unwrap(), rotation_z_val.unwrap(),rotation_w_val.unwrap()];
            let offset_vec = vec![offset_x_val.unwrap(), offset_y_val.unwrap(), offset_z_val.unwrap()];

            let (x, y, z) = map_to_ecef_elementwise(map_vec, rotation_vec, offset_vec);

            ecef_x.append_value(x);
            ecef_y.append_value(y);
            ecef_z.append_value(z);
        } 

    let ser_ecef_x = ecef_x.finish().into_series();
    let ser_ecef_y = ecef_y.finish().into_series();
    let ser_ecef_z = ecef_z.finish().into_series();

    let out_chunked = StructChunked::new("ecef", &[ser_ecef_x, ser_ecef_y, ser_ecef_z])?;
    Ok(out_chunked.into_series())

}

fn map_output(_: &[Field]) -> PolarsResult<Field> {
    let v: Vec<Field> = vec![
        Field::new("x", DataType::Float64),
        Field::new("y", DataType::Float64),
        Field::new("z", DataType::Float64),
    ];
    Ok(Field::new("map", DataType::Struct(v)))
}


#[polars_expr(output_type_func=map_output)]
fn ecef_to_map(inputs: &[Series]) -> PolarsResult<Series> {

    let ca = inputs[0].struct_()?;
    let rotation_ca = inputs[1].struct_()?;
    let offset_ca = inputs[2].struct_()?;

    let ecef_x_ser = ca.field_by_name("x")?;
    let ecef_y_ser = ca.field_by_name("y")?;
    let ecef_z_ser = ca.field_by_name("z")?;
    let rotation_x_ser = rotation_ca.field_by_name("x")?;
    let rotation_y_ser = rotation_ca.field_by_name("y")?;
    let rotation_z_ser = rotation_ca.field_by_name("z")?;
    let rotation_w_ser = rotation_ca.field_by_name("w")?;
    let offset_x_ser = offset_ca.field_by_name("x")?;
    let offset_y_ser = offset_ca.field_by_name("y")?;
    let offset_z_ser = offset_ca.field_by_name("z")?;

    let ecef_x: &ChunkedArray<Float64Type>= ecef_x_ser.f64()?;
    let ecef_y: &ChunkedArray<Float64Type>= ecef_y_ser.f64()?;
    let ecef_z: &ChunkedArray<Float64Type>= ecef_z_ser.f64()?;
    let rotation_x: &ChunkedArray<Float64Type>= rotation_x_ser.f64()?;
    let rotation_y: &ChunkedArray<Float64Type>= rotation_y_ser.f64()?;
    let rotation_z: &ChunkedArray<Float64Type>= rotation_z_ser.f64()?;
    let rotation_w: &ChunkedArray<Float64Type>= rotation_w_ser.f64()?;
    let offset_x: &ChunkedArray<Float64Type>= offset_x_ser.f64()?;
    let offset_y: &ChunkedArray<Float64Type>= offset_y_ser.f64()?;
    let offset_z: &ChunkedArray<Float64Type>= offset_z_ser.f64()?;

    let mut map_x: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("x", ca.len());
    let mut map_y: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("y", ca.len());
    let mut map_z: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("z", ca.len());

    for (ecef_x_val, ecef_y_val, ecef_z_val, rotation_x_val, rotation_y_val, rotation_z_val, rotation_w_val, offset_x_val, offset_y_val, offset_z_val) 
        in izip!(ecef_x, ecef_y, ecef_z, rotation_x, rotation_y, rotation_z, rotation_w, offset_x, offset_y, offset_z) {
            let ecef_vec = vec![ecef_x_val.unwrap(), ecef_y_val.unwrap(), ecef_z_val.unwrap()];
            let rotation_vec = vec![rotation_x_val.unwrap(), rotation_y_val.unwrap(), rotation_z_val.unwrap(),rotation_w_val.unwrap()];
            let offset_vec = vec![offset_x_val.unwrap(), offset_y_val.unwrap(), offset_z_val.unwrap()];

            let (x, y, z) = ecef_to_map_elementwise(ecef_vec, rotation_vec, offset_vec);

            map_x.append_value(x);
            map_y.append_value(y);
            map_z.append_value(z);
        } 

    let ser_map_x = map_x.finish().into_series();
    let ser_map_y = map_y.finish().into_series();
    let ser_map_z = map_z.finish().into_series();

    let out_chunked = StructChunked::new("map", &[ser_map_x, ser_map_y, ser_map_z])?;
    Ok(out_chunked.into_series())

}


fn lla_output(_: &[Field]) -> PolarsResult<Field> {
    let v: Vec<Field> = vec![
        Field::new("lon", DataType::Float64),
        Field::new("lat", DataType::Float64),
        Field::new("alt", DataType::Float64),
    ];
    Ok(Field::new("coordinates", DataType::Struct(v)))
}

#[polars_expr(output_type_func=lla_output)]
fn ecef_to_lla(inputs: &[Series]) -> PolarsResult<Series> {

    let ca = inputs[0].struct_()?;

    let ecef_x_ser = ca.field_by_name("x")?;
    let ecef_y_ser = ca.field_by_name("y")?;
    let ecef_z_ser = ca.field_by_name("z")?;

    let ecef_x = ecef_x_ser.f64()?;
    let ecef_y = ecef_y_ser.f64()?;
    let ecef_z = ecef_z_ser.f64()?;


    let mut longitude: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("lon", ca.len());
    let mut latitude: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("lat", ca.len());
    let mut altitude: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("alt", ca.len());

    for (x, y, z) in izip!(ecef_x.into_iter(), ecef_y.into_iter(), ecef_z.into_iter()) {
        match (x, y, z) {
            (Some(x), Some(y), Some(z)) => {
                let (lon, lat, alt) = ecef_to_lla_elementwise(x, y, z);
                longitude.append_value(lon);
                latitude.append_value(lat);
                altitude.append_value(alt);
            },
            _ => {
                longitude.append_null();
                latitude.append_null();
                altitude.append_null();
            }
        }
    }

    let ser_lon = longitude.finish().into_series();
    let ser_lat = latitude.finish().into_series();
    let ser_alt = altitude.finish().into_series();

    let out_chunked = StructChunked::new("coordinates", &[ser_lon, ser_lat, ser_alt])?;
    Ok(out_chunked.into_series())
    

}


#[polars_expr(output_type_func=ecef_output)]
fn lla_to_ecef(inputs: &[Series]) -> PolarsResult<Series> {

    let ca = inputs[0].struct_()?;

    let lon_ser = ca.field_by_name("lon")?;
    let lat_ser = ca.field_by_name("lat")?;
    let alt_ser = ca.field_by_name("alt")?;

    let lon = lon_ser.f64()?;
    let lat = lat_ser.f64()?;
    let alt = alt_ser.f64()?;


    let mut ecef_x: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("x", ca.len());
    let mut ecef_y: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("y", ca.len());
    let mut ecef_z: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("z", ca.len());

    for (lon_op, lat_op, alt_op) in izip!(lon.into_iter(), lat.into_iter(), alt.into_iter()) {
        match (lon_op, lat_op, alt_op) {
            (Some(lo), Some(la), Some(al)) => {
                let (x, y, z) = lla_to_ecef_elementwise(lo, la, al);
                ecef_x.append_value(x);
                ecef_y.append_value(y);
                ecef_z.append_value(z);
            },
            _ => {
                ecef_x.append_null();
                ecef_y.append_null();
                ecef_z.append_null();
            }
        }
    }

    let ser_x = ecef_x.finish().into_series();
    let ser_y = ecef_y.finish().into_series();
    let ser_z = ecef_z.finish().into_series();

    let out_chunked = StructChunked::new("coordinates", &[ser_x, ser_y, ser_z])?;
    Ok(out_chunked.into_series())
    

}

