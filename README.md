# The Project

This Polars plugin provides functionality which can be loosely described as tranformation of coordinates and extraction of features from them.

 It contains functions which were needed in personal and work projects, therefore its set of features might appear a bit random. Nevertheless one can find it useful in projects related to robotics, geospatial science, spatial analytics etc. 

The functions are divided among three namespaces: `transform`, `s2`, `distance`:

- `transform` namespace contains functions for converting coordinates from\to map, ecef, lla, utm reference frames.

- `s2` namespace contains functions which allow to work with [S2 Cells](http://s2geometry.io/about/overview)

- `distance` namespace allows to calculate distances between coordinates.


# Getting Started

### Installation

```
pip install poalrs-coord-transforms
```

### Usage 

```
import polars_coord_transforms
```

In order to use plugin, coordinates should be represented as `struct` with fields `x`, `y`, `z` (or, in case of LLA-points: `lon`, `lat`, `alt`)!

For instance, if coordinates are in separate columns, one can make a valid `struct` with `pl.struct` native Polars function:

```
import polars as pl

df = pl.DataFrame(
    dict(
            lon=[31.409197919000064,],
            lat=[58.860667429000046,],
            alt=[57.309668855211015,],
        )
)

df.with_columns(
    point=pl.struct("lon", "lat", "alt")
)
```

### Examples


#### `transform`

```
df.with_columns(
    ecef=pl.col("pose").transform.map_to_ecef(
        pl.col("rotation"), pl.col("offset")
    )
).head()
```


```
df.with_columns(
    lla=pl.col("ecef").transform.ecef_to_lla()
)
df.head()
```


```
df.with_columns(
    utm=pl.col("lla").transform.lla_to_utm()
)
df.head()
```

#### `s2`

```
df.with_columns(
    cellid_30=pl.col("lla").s2.lonlat_to_cellid(level=30),
    cellid_28=pl.col("lla").s2.lonlat_to_cellid(level=28),
    cellid_5=pl.col("lla").s2.lonlat_to_cellid(level=5),
).head()
```

```
df.with_columns(
    is_in_cell=pl.col("cellid").s2.cell_contains_point(pl.col("lla_points"))
)
```

#### `distance`

```
df.with_columns(
    distance=pl.col("point_1").distance.euclidean_3d(pl.col("point_2"))
)
```


```
df.with_columns(
    cosine_sim=pl.col("point_1").distance.cosine_similarity_3d(pl.col("point_2"))
)
```


