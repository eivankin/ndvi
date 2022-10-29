from typing import Optional
from enum import Enum

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import json
from sklearn.preprocessing import LabelEncoder, FunctionTransformer

label_encoder = None


class NDVIClasses(Enum):
    NONE = 0.0
    DRY = 0.2
    MODERATE = 0.4
    WET = 0.6
    EXTREMELY_WET = 1

    @classmethod
    def encode_ndvi(cls, ndvi: float) -> int:
        for i, c in enumerate(cls):
            if ndvi < c.value:
                return i
        return 5


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return replace_zeros_with_mean(sort_cols(df))


def sort_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df[sorted(df.columns, key=lambda x: tuple(x.split("_")))]


def replace_zeros_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    min_idx = None
    tmp.loc[:, tmp.columns.str.contains("nd_mean_")] = tmp.loc[:, tmp.columns.str.contains(
        "nd_mean_")].replace(0, np.nan)
    for i, col in enumerate(tmp):
        if col.startswith("nd_mean_"):
            if min_idx is None:
                min_idx = i
                tmp.iloc[:, i] = tmp.iloc[:, i].fillna(tmp.iloc[:, i + 1].mean())
            else:
                tmp.iloc[:, i] = tmp.iloc[:, i].fillna(tmp.iloc[:, i - 1:i + 2:2].mean(axis=1))
    return tmp.set_index("id")


def df_to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(df.drop([".geo"], axis=1))
    gdf["geometry"] = df[".geo"].apply(lambda x: shape(json.loads(x)))
    return gdf


def add_region(df: gpd.GeoDataFrame, regions: gpd.GeoDataFrame):
    """Adds region column inplace"""
    df["region"] = df.apply(lambda row: regions.loc[
        regions.geometry.contains(row["geometry"].centroid), "name_en"].ravel()[0], axis=1)


def get_label_encoder(df: Optional[pd.DataFrame] = None) -> Optional[LabelEncoder]:
    global label_encoder
    if label_encoder is None and df is not None:
        label_encoder = LabelEncoder()
        label_encoder.fit(df)
    return label_encoder


def encode_region(df: pd.DataFrame):
    """Replaces region's name with index inplace"""
    df["region"] = get_label_encoder(df["region"]).transform(df["region"])


def drop_unused_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(["area", "geometry"], axis=1)


def preprocess_for_tsfresh(df: pd.DataFrame, train=True):
    tmp = df.reset_index()
    tmp = tmp.rename(columns={"index": "id"})
    target = None
    if train:
        target = tmp[["id", "crop"]]
        target = target.set_index("id")
        target = pd.Series(data=target["crop"], index=target.index)
        data = tmp.drop(columns="crop")
    else:
        data = tmp.copy()

    nd_cols = get_ndvi_columns(df)

    tmp = data[["id"] + nd_cols].melt(id_vars=["id"], var_name="time",
                                      value_vars=nd_cols,
                                      value_name="ndvi")
    tmp["time"] = [nd_cols.index(c) for c in tmp["time"]]

    return target, tmp


def get_ndvi_transformer() -> FunctionTransformer:
    return FunctionTransformer(check_inverse=False,
                               func=lambda i: i.applymap(NDVIClasses.encode_ndvi))


def get_ndvi_columns(df: pd.DataFrame) -> list[str]:
    return list(df.loc[:, df.columns.str.contains("nd_")].columns)


def merge_mean(df: pd.DataFrame, next_col_iter: callable) -> pd.DataFrame:
    nd_cols = get_ndvi_columns(df)
    tmp = df.drop(columns=nd_cols)

    curr_idx = 0
    curr_col = nd_cols[0]
    next_col = next_col_iter(nd_cols, curr_col)
    while curr_col is not None:
        tmp[f"nd_mean_{curr_idx}"] = df.loc[:, curr_col:next_col].mean(axis=1)
        curr_col = next_col
        next_col = next_col_iter(curr_col)
        curr_idx += 1
    return tmp


def half_month(cols: list[str], curr: str):
    next_col = None

    def dm(col):
        return tuple(map(int, col.split("-")[-2:]))[::-1]

    day_curr, month_curr = dm(curr)

    if day_curr >= 15:
        for c in cols:
            if dm(c)[1] > month_curr:
                next_col = c
                break
    else:
        for c in cols:
            d, m = dm(c)
            if d >= 15 and m == month_curr:
                next_col = c
                break

    if next_col is None and cols.index(curr) != len(cols) - 1:
        return cols[-1]

    return next_col
