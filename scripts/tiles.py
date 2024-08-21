import xarray
import numpy as np
import os
import yaml
from rasterio.enums import Resampling
from utils import BandsOptions, DatasetOptions
import pandas as pd
from datetime import date

def create_tiles(
    source_path,
    size,
    step_size,
    timestamp: list | None = None,
    reshape_size: int | None = None,
    nodata_allowed: float = 0.1,
    water_threshold: float = 0.1,
    save_change_mask: bool = False,
    save_water_mask: bool = False,
    bands: BandsOptions = BandsOptions.NDWI,
    target_path: str = None,
    target_filename: str = "features.nc",
):
    nodata_allowed = 1 - nodata_allowed
    res = []
    area = size * size
    features = xarray.open_dataset(
        os.path.join(source_path, "features.nc"), decode_coords="all"
    )[bands.get_bands(expand_indexes=True)]
    if reshape_size is not None:
        features = features.rio.reproject(
            features.rio.crs,
            shape=(reshape_size, reshape_size),
            resampling=Resampling.bilinear,
        )

    full_labels = xarray.open_dataset(
        os.path.join(source_path, "labels.nc"), decode_coords="all"
    )
    if reshape_size is not None:
        full_labels = full_labels.rio.reproject(
            full_labels.rio.crs,
            shape=(reshape_size, reshape_size),
            resampling=Resampling.bilinear,
        )
    crs = features.rio.crs.to_string()
    features = features.where(features != 0, np.nan)
    time = features["t"]
    labels_change = full_labels["water_change"]
    labels_water = full_labels["start"]
    if isinstance(features, xarray.Dataset):
        features = features.to_dataarray()
    features = features.squeeze()
    h, w = features.shape[-2:]
    if target_path is None:
        target_path = source_path
    if not os.path.exists(os.path.join(target_path, "tiles")):
        os.mkdir(os.path.join(target_path, "tiles"))
    step = 0
    depth = int(features.shape[0])
    if features.ndim > 3:
        depth = depth * int(features.shape[1])
    for i in range(0, h, step_size):
        for j in range(0, w, step_size):
            if features.ndim == 3:
                cut = features[:, i : i + size, j : j + size]
            else:
                cut = features[:, :, i : i + size, j : j + size].squeeze()
            if labels_water.ndim == 3:
                labels_water_cut = labels_water[:, i : i + size, j : j + size].squeeze()
                labels_change_cut = labels_change[:, i : i + size, j : j + size].squeeze()
            else:
                labels_water_cut = labels_water[i : i + size, j : j + size].squeeze()
                labels_change_cut = labels_change[i : i + size, j : j + size].squeeze()
            if labels_change_cut.shape[-2:] != (step_size,step_size):
                continue
            ratio = labels_water_cut.where(labels_water_cut == 1).count().values / area
            not_nulls = cut.notnull().sum().values / depth
            step += 1
            if ratio > water_threshold and ratio < 1-water_threshold and not_nulls / area > nodata_allowed:
                change_value = labels_change_cut.sum().values / area
                res.append(
                    {
                        "features": cut.where(lambda arr: arr.notnull(), 0),
                        "change": change_value,
                        "mask": labels_change_cut if save_change_mask else None,
                        "water" : labels_water_cut if save_water_mask else None,
                        "ratio": ratio,
                    }
                )
    for i, data in enumerate(res):
        tile = data["features"]
        folder = os.path.join(target_path, "tiles", "tile_" + str(i))
        if not os.path.exists(folder):
            os.mkdir(folder)
        tile.to_netcdf(os.path.join(folder, target_filename))
        tile.close()
        if save_change_mask:
            mask = data["mask"]
            mask.to_netcdf(os.path.join(folder, "mask.nc"))
            mask.close()
        if save_water_mask:
            water = data["water"]
            water.to_netcdf(os.path.join(folder, "water.nc"))
            water.close()
        metadata = {
            "crs": crs,
            "change": float(data["change"]),  # yaml can't serialize numpy types
            "time": (
                timestamp
                if timestamp is not None
                else [time[0].values, time[-1].values]
            ),
            "water_ratio": float(data["ratio"])
        }
        with open(os.path.join(folder, "metadata.yaml"), "w") as f:
            yaml.safe_dump(metadata, f)
    features.close()
    full_labels.close()
    return len(res)


def multi_tiles(
    source_folder: str,
    row: pd.Series,
    patch_name: str,
    target_datasets: list[DatasetOptions],
    target_timestamp: list[date],
    nodata_allowed: float = 0.1,
    save_change_mask: bool = False,
    save_water_mask: bool = False,
):
    for dataset in target_datasets:
        target_csv = dataset.target_csv
        target_folder = os.path.join(dataset.target_folder,patch_name)
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
        target_df = pd.read_csv(target_csv,header=0,usecols=["coordinates","name","category","date","downloaded","split","num_patches","region"],index_col=["name"],sep=",")
        if patch_name in target_df.index:
            continue
        tiles_number = create_tiles(
            source_folder,
            dataset.tile_size,
            dataset.step_size,
            timestamp=[target_timestamp[0], target_timestamp[1]],
            reshape_size=None,
            nodata_allowed=nodata_allowed,
            save_change_mask=save_change_mask,
            save_water_mask=save_water_mask,
            bands=dataset.bands,
            target_path=target_folder,
        )
        if tiles_number is not None and tiles_number > 0:
            with open (target_csv, "ab") as f:
                f.write(f"{row['coordinates']},{patch_name},{row['category']},{target_timestamp[0]},1,{row['split']},{row['region']},{tiles_number}\n".encode('utf-8'))
        else:
            return False
    return True