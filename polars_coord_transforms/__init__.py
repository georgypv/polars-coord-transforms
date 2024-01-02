import polars as pl
from polars.utils.udfs import _get_shared_lib_location
from polars.type_aliases import PolarsDataType

from typing import Protocol, Iterable, cast


lib = _get_shared_lib_location(__file__)

@pl.api.register_expr_namespace("s2")
class S2NameSpace:

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def lonlat_to_cellid(self, level: int = 30) -> pl.Expr:

        if level < 1 or level > 30:
            raise ValueError("`level` parameter must be between 1 and 30!")
        
        return self._expr.register_plugin(
            lib=lib,
            symbol="lonlat_to_cellid",
            is_elementwise=True,
            kwargs={"level": level}
        )
    
    def cellid_to_lonlat(self) -> pl.Expr:
        return self._expr.register_plugin(
            lib=lib,
            symbol="cellid_to_lonlat",
            is_elementwise=True
        )

    def cell_contains_point(self, point: pl.Expr) -> pl.Expr:
        return self._expr.register_plugin(
            lib=lib,
            symbol="cell_contains_point",
            is_elementwise=True,
            args=[point]
        ) 
    
    def cellid_to_vertices(self) -> pl.Expr:
        return self._expr.register_plugin(
            lib=lib,
            symbol="cellid_to_vertices",
            is_elementwise=True,
        )
    

@pl.api.register_expr_namespace("transform")
class TransformNameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def enu_to_ecef(self, rotation: pl.Expr, offset: pl.Expr) -> pl.Expr:
        return self._expr.register_plugin(
            lib=lib,
            symbol="enu_to_ecef",
            is_elementwise=True,
            args=[rotation, offset]
        )
    
    def ecef_to_enu(self, rotation: pl.Expr, offset: pl.Expr) -> pl.Expr:
        return self._expr.register_plugin(
            lib=lib,
            symbol="ecef_to_enu",
            is_elementwise=True,
            args=[rotation, offset]
        )
    
    
    def ecef_to_lla(self) -> pl.Expr:
        return self._expr.register_plugin(
            lib=lib,
            symbol="ecef_to_lla",
            is_elementwise=True,
        )
    
    def lla_to_ecef(self) -> pl.Expr:
        return self._expr.register_plugin(
            lib=lib,
            symbol="lla_to_ecef",
            is_elementwise=True,
        )
    

class CoordTransformExpr(pl.Expr):
    @property
    def s2(self) -> S2NameSpace:
        return S2NameSpace(self)

    @property
    def transform(self) -> TransformNameSpace:
        return TransformNameSpace(self)
    

class CTColumn(Protocol):
    def __cal__(
            self,
            name:  str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType],
            *more_names: str | PolarsDataType,
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


col = cast(CTColumn, pl.col)

__all__ = [
    "col"
]
