use itertools::Itertools;
use polars::prelude::*;
use polars::datatypes::DataType;
use pyo3_polars::derive::polars_expr;

use itertools::izip;
use serde::Deserialize;

use crate::s2_functions::*;
use crate::coord_transforms::*;


fn apply_rotation_to_map(
    coords_ca: &StructChunked,
    rotation_ca: &StructChunked,
    offset_ca: &StructChunked,
    result_struct_name: &str,
    func_elementwise: impl Fn(Vec<f64>, Vec<f64>, Vec<f64>) -> (f64, f64, f64)
    ) -> Result<StructChunked, PolarsError> {
    
        let x_ser = coords_ca.field_by_name("x").unwrap();
        let y_ser = coords_ca.field_by_name("y").unwrap();
        let z_ser = coords_ca.field_by_name("z").unwrap();
        let rotation_x_ser = rotation_ca.field_by_name("x").unwrap();
        let rotation_y_ser = rotation_ca.field_by_name("y").unwrap();
        let rotation_z_ser = rotation_ca.field_by_name("z").unwrap();
        let rotation_w_ser = rotation_ca.field_by_name("w").unwrap();
        let offset_x_ser = offset_ca.field_by_name("x").unwrap();
        let offset_y_ser = offset_ca.field_by_name("y").unwrap();
        let offset_z_ser = offset_ca.field_by_name("z").unwrap();
    
        let x_ca: &ChunkedArray<Float64Type>= x_ser.f64().unwrap();
        let y_ca: &ChunkedArray<Float64Type>= y_ser.f64().unwrap();
        let z_ca: &ChunkedArray<Float64Type>= z_ser.f64().unwrap();
        let rotation_x: &ChunkedArray<Float64Type>= rotation_x_ser.f64().unwrap();
        let rotation_y: &ChunkedArray<Float64Type>= rotation_y_ser.f64().unwrap();
        let rotation_z: &ChunkedArray<Float64Type>= rotation_z_ser.f64().unwrap();
        let rotation_w: &ChunkedArray<Float64Type>= rotation_w_ser.f64().unwrap();
        let offset_x: &ChunkedArray<Float64Type>= offset_x_ser.f64().unwrap();
        let offset_y: &ChunkedArray<Float64Type>= offset_y_ser.f64().unwrap();
        let offset_z: &ChunkedArray<Float64Type>= offset_z_ser.f64().unwrap();
    
        let mut x_cb: PrimitiveChunkedBuilder<Float64Type> =
            PrimitiveChunkedBuilder::new("x", coords_ca.len());
        let mut y_cb: PrimitiveChunkedBuilder<Float64Type> =
            PrimitiveChunkedBuilder::new("y", coords_ca.len());
        let mut z_cb: PrimitiveChunkedBuilder<Float64Type> =
            PrimitiveChunkedBuilder::new("z", coords_ca.len());
        for (x_val, y_val, z_val, rotation_x_val, rotation_y_val, rotation_z_val, rotation_w_val, offset_x_val, offset_y_val, offset_z_val) 
            in izip!(x_ca, y_ca, z_ca, rotation_x, rotation_y, rotation_z, rotation_w, offset_x, offset_y, offset_z) {
                let map_vec = vec![x_val.unwrap(), y_val.unwrap(), z_val.unwrap()];
                let rotation_vec = vec![rotation_x_val.unwrap(), rotation_y_val.unwrap(), rotation_z_val.unwrap(),rotation_w_val.unwrap()];
                let offset_vec = vec![offset_x_val.unwrap(), offset_y_val.unwrap(), offset_z_val.unwrap()];
    
                let (x, y, z) = func_elementwise(map_vec, rotation_vec, offset_vec);
    
                x_cb.append_value(x);
                y_cb.append_value(y);
                z_cb.append_value(z);
            } 
    
        let ser_out_x = x_cb.finish().into_series();
        let ser_out_y = y_cb.finish().into_series();
        let ser_out_z = z_cb.finish().into_series();
    
        let out_chunked = StructChunked::new(result_struct_name, &[ser_out_x, ser_out_y, ser_out_z]);
        out_chunked
    
}


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

    let out_chunked = apply_rotation_to_map(map_ca, rotation_ca, offset_ca, "ecef", map_to_ecef_elementwise);

    Ok(out_chunked?.into_series())

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

    let ecef_ca = inputs[0].struct_()?;
    let rotation_ca = inputs[1].struct_()?;
    let offset_ca = inputs[2].struct_()?;

    let out_chunked = apply_rotation_to_map(ecef_ca, rotation_ca, offset_ca, "map", ecef_to_map_elementwise);
    Ok(out_chunked?.into_series())

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


#[polars_expr(output_type_func=map_output)]
fn rotate_map_coords(inputs: &[Series]) -> PolarsResult<Series> {

    let map_ca = inputs[0].struct_()?;
    let rotation_ca = inputs[1].struct_()?;
    let scale_ca = inputs[2].struct_()?;

    let out_chunked = apply_rotation_to_map(map_ca, rotation_ca, scale_ca, "map", rotate_map_coords_elementwise);
    
    Ok(out_chunked?.into_series())

}

