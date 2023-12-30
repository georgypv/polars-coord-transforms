use polars::prelude::*;
use polars::datatypes::DataType;
use pyo3_polars::derive::{polars_expr};

use itertools::izip;
use serde::Deserialize;

use crate::s2_functions::{cellid_to_lonlat_elementwise, lonlat_to_cellid_elementwise};
use crate::coord_transforms::{enu_to_ecef_elementwise, ecef_to_lla_elementwise};



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

fn enu_to_ecef_output(_: &[Field]) -> PolarsResult<Field> {
    let v: Vec<Field> = vec![
        Field::new("x", DataType::Float64),
        Field::new("y", DataType::Float64),
        Field::new("z", DataType::Float64),
    ];
    Ok(Field::new("ecef", DataType::Struct(v)))
}

#[polars_expr(output_type_func=enu_to_ecef_output)]
fn enu_to_ecef(inputs: &[Series]) -> PolarsResult<Series> {

    let enu_ca = inputs[0].struct_()?;
    let rotation_ca = inputs[1].struct_()?;
    let offset_ca = inputs[2].struct_()?;

    let enu_x_ser = enu_ca.field_by_name("x")?;
    let enu_y_ser = enu_ca.field_by_name("y")?;
    let enu_z_ser = enu_ca.field_by_name("z")?;
    let rotation_x_ser = rotation_ca.field_by_name("x")?;
    let rotation_y_ser = rotation_ca.field_by_name("y")?;
    let rotation_z_ser = rotation_ca.field_by_name("z")?;
    let rotation_w_ser = rotation_ca.field_by_name("w")?;
    let offset_x_ser = offset_ca.field_by_name("x")?;
    let offset_y_ser = offset_ca.field_by_name("y")?;
    let offset_z_ser = offset_ca.field_by_name("z")?;

    let enu_x: &ChunkedArray<Float64Type>= enu_x_ser.f64()?;
    let enu_y: &ChunkedArray<Float64Type>= enu_y_ser.f64()?;
    let enu_z: &ChunkedArray<Float64Type>= enu_z_ser.f64()?;
    let rotation_x: &ChunkedArray<Float64Type>= rotation_x_ser.f64()?;
    let rotation_y: &ChunkedArray<Float64Type>= rotation_y_ser.f64()?;
    let rotation_z: &ChunkedArray<Float64Type>= rotation_z_ser.f64()?;
    let rotation_w: &ChunkedArray<Float64Type>= rotation_w_ser.f64()?;
    let offset_x: &ChunkedArray<Float64Type>= offset_x_ser.f64()?;
    let offset_y: &ChunkedArray<Float64Type>= offset_y_ser.f64()?;
    let offset_z: &ChunkedArray<Float64Type>= offset_z_ser.f64()?;

    let mut ecef_x: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("x", enu_ca.len());
    let mut ecef_y: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("y", enu_ca.len());
    let mut ecef_z: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("z", enu_ca.len());

    for (enu_x_val, enu_y_val, enu_z_val, rotation_x_val, rotation_y_val, rotation_z_val, rotation_w_val, offset_x_val, offset_y_val, offset_z_val) 
        in izip!(enu_x, enu_y, enu_z, rotation_x, rotation_y, rotation_z, rotation_w, offset_x, offset_y, offset_z) {
            let enu_vec = vec![enu_x_val.unwrap(), enu_y_val.unwrap(), enu_z_val.unwrap()];
            let rotation_vec = vec![rotation_x_val.unwrap(), rotation_y_val.unwrap(), rotation_z_val.unwrap(),rotation_w_val.unwrap()];
            let offset_vec = vec![offset_x_val.unwrap(), offset_y_val.unwrap(), offset_z_val.unwrap()];

            let (x, y, z) = enu_to_ecef_elementwise(enu_vec, rotation_vec, offset_vec);

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


fn ecef_to_lla_output(_: &[Field]) -> PolarsResult<Field> {
    let v: Vec<Field> = vec![
        Field::new("lon", DataType::Float64),
        Field::new("lat", DataType::Float64),
        Field::new("alt", DataType::Float64),
    ];
    Ok(Field::new("coordinates", DataType::Struct(v)))
}

#[polars_expr(output_type_func=ecef_to_lla_output)]
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