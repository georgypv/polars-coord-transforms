from pathlib import Path
import polars as pl
from polars.plugins import register_plugin_function
from polars.type_aliases import PolarsDataType

from typing import Protocol, Iterable, cast


@pl.api.register_expr_namespace("s2")
class S2NameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def lonlat_to_cellid(self, level: int = 30) -> pl.Expr:
        if level < 1 or level > 30:
            raise ValueError("`level` parameter must be between 1 and 30!")
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="lonlat_to_cellid",
            args=self._expr,
            kwargs={"level": level},
            is_elementwise=True
        )

    def cellid_to_lonlat(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="cellid_to_lonlat",
            args=self._expr,
            is_elementwise=True
        )

    def cell_contains_point(self, point: pl.Expr) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="cell_contains_point",
            args=[self._expr, point],
            is_elementwise=True
        )

    def cellid_to_vertices(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="cellid_to_vertices",
            args=self._expr,
            is_elementwise=True
        )


@pl.api.register_expr_namespace("transform")
class TransformNameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def map_to_ecef(self, rotation: pl.Expr, offset: pl.Expr) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="map_to_ecef",
            args=[self._expr, rotation, offset],
            is_elementwise=True
        )

    def ecef_to_map(self, rotation: pl.Expr, offset: pl.Expr) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="ecef_to_map",
            args=[self._expr, rotation, offset],
            is_elementwise=True
        )

    def ecef_to_lla(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="ecef_to_lla",
            args=self._expr,
            is_elementwise=True
        )

    def lla_to_ecef(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="lla_to_ecef",
            args=self._expr,
            is_elementwise=True
        )
    
    def lla_to_utm(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="lla_to_utm",
            args=self._expr,
            is_elementwise=True
        )
    
    def lla_to_utm_zone_number(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="lla_to_utm_zone_number",
            args=self._expr,
            is_elementwise=True
        )

    def rotate_map_coords(self, rotation: pl.Expr, scale: pl.Expr) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="rotate_map_coords",
            args=[self._expr, rotation, scale],
            is_elementwise=True
        )

    
    def interpolate_linear(self, other: pl.Expr, coef=0.5):
        if coef < 0 or coef > 1:
            raise ValueError("`coef` parameter must be between 0 and 1!")

        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="interpolate_linear",
            args=[self._expr, other],
            kwargs={"coef": coef},
        )
    
    def quat_to_euler_angles(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="quat_to_euler_angles",
            args=self._expr,
            is_elementwise=True
        )
    
    def get_rotation_matrix(self, offset: pl.Expr) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="get_rotation_matrix",
            args=[self._expr, offset],
            is_elementwise=True
        )

@pl.api.register_expr_namespace("distance")
class DistanceNameSpace:

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def euclidean_3d(self, other: pl.Expr) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="euclidean_3d",
            args=[self._expr, other],
            is_elementwise=True

        )
    

    def euclidean_2d(self, other: pl.Expr) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="euclidean_2d",
            args=[self._expr, other],
            is_elementwise=True

        )
    
    def cosine_similarity_2d(self, other: pl.Expr) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="cosine_similarity_2d",
            args=[self._expr, other],
            is_elementwise=True
        )
    
    def cosine_similarity_3d(self, other: pl.Expr) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="cosine_similarity_3d",
            args=[self._expr, other],
            is_elementwise=True
        )


class CoordTransformExpr(pl.Expr):
    @property
    def s2(self) -> S2NameSpace:
        return S2NameSpace(self)

    @property
    def transform(self) -> TransformNameSpace:
        return TransformNameSpace(self)
    
    @property
    def distance(self) -> DistanceNameSpace:
        return TransformNameSpace(self)


class CTColumn(Protocol):
    def __cal__(
        self,
        name,
        *more_names,
    ) -> CoordTransformExpr:
        ...

    def __getattr__(self, name: str) -> pl.Expr:
        ...

    @property
    def s2(self) -> S2NameSpace:
        ...

    @property
    def transform(self) -> TransformNameSpace:
        ...

    @property
    def distance(self) -> DistanceNameSpace:
        ...


col = cast(CTColumn, pl.col)

__all__ = ["col"]
