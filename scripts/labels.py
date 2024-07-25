import os
from datetime import date, timedelta
from pathlib import Path
from typing import List
from shapely.geometry import Point
import openeo
import xarray
import rioxarray
import numpy as np
from rasterio.enums import Resampling
from water_mask import download_water_mask, download_water_mask_monthly, download_water_mask_yearly, WaterMaskType

def old_generate_change_labels(event_location: Point, starting_event_date: date, event_date: date, folder_path: str, output_path: str, clear_temp_data: bool = True):
    first_labels_path = download_water_mask(
        event_location,
        starting_event_date,
        folder_path,
        "water_mask_{}.tif".format(str(starting_event_date.year)),
        offset=0.2,
    )
    last_labels_path = download_water_mask(
        event_location,
        event_date,
        folder_path,
        "water_mask_{}.tif".format(str(event_date.year)),
        offset=0.2,
    )
    print("First labels path: ", first_labels_path)
    print("Last labels path: ", last_labels_path)
    data = rioxarray.open_rasterio(first_labels_path)
    start = xarray.where(data >= 2, 1, 0)
    data_2 = rioxarray.open_rasterio(last_labels_path)
    finish = xarray.where(data_2 >= 2, 1, 0)
    diff = xarray.DataArray(data=(start - finish),coords=start.coords,dims=start.dims)
    diff = xarray.where(diff != 0, xarray.where(diff > 0, 1, -1), 0)
    diff : xarray.Dataset = xarray.Dataset(
        {"water_change": diff, "start": start, "finish": finish}, coords=data.coords
    )
    features : xarray.Dataset = rioxarray.open_rasterio(output_path).rio.reproject(dst_crs=data.rio.crs,resampling=Resampling.bilinear)
    diff = diff.rio.clip_box(*features.rio.bounds(), crs=data.rio.crs)
    diff = diff.rio.reproject(
        dst_crs=data.rio.crs,
        shape=features.to_dataarray().shape[-2:] if isinstance(features, xarray.Dataset) else features.shape[-2:], # type: ignore
        resampling=Resampling.bilinear,
    )
    diff.coords["x"] = features.coords["x"]
    diff.coords["y"] = features.coords["y"]
    diff.to_netcdf(os.path.join(folder_path, "labels.nc"))
    diff.close()
    data.close()
    data_2.close()
    features.to_netcdf(os.path.join(folder_path, "features.nc"))
    features.close()
    if clear_temp_data:
        os.remove(first_labels_path)
        os.remove(last_labels_path)
        os.remove(output_path)

def old_generate_monthly_labels(event_location: Point, starting_event_date: date, event_date: date, folder_path: str, output_path: str, clear_temp_data: bool = True, offset: float = 0.1):
    offset=offset+0.1
    first_label_path = download_water_mask_monthly(
        event_location,
        starting_event_date,
        folder_path,
        "water_mask_{}.tif".format(str(starting_event_date.year)),
        offset=offset,
    )
    first_label = rioxarray.open_rasterio(first_label_path).squeeze()
    start_date = starting_event_date
    first_label_tries=0
    base_crs = first_label.rio.crs
    while ((first_label == 0).sum().values / first_label.count().values) > 0.01:
        start_date = start_date - timedelta(days=30)
        out = download_water_mask_monthly(
            event_location,
            start_date,
            folder_path,
            f"first_temp_{first_label_tries}.tif",
            offset=offset,
        )
        temp = rioxarray.open_rasterio(out).squeeze()
        first_label = xarray.where(first_label == 0, temp, first_label)
        temp.close()
        os.remove(out)
        first_label_tries += 1
        if first_label_tries > 4:
            if np.max(first_label) < 1:
                raise Exception("Error generating first label")
            else:
                print("Warning: low quality data for first label")
                break
    last_label_path = download_water_mask_monthly(
        event_location,
        event_date,
        folder_path,
        "water_mask_{}.tif".format(str(event_date.year)),
        offset=offset,
    )
    last_label = rioxarray.open_rasterio(last_label_path).squeeze()
    last_label_tries=0
    while ((last_label == 0).sum().values / last_label.count().values) > 0.01:
        event_date = event_date + timedelta(days=30)
        out = download_water_mask_monthly(
            event_location,
            event_date,
            folder_path,
            f"last_temp_{last_label_tries}.tif",
            offset=offset,
        )
        temp = rioxarray.open_rasterio(out).squeeze()
        last_label = xarray.where(last_label == 0, temp, last_label)
        temp.close()
        os.remove(out)
        last_label_tries += 1
        if last_label_tries > 4:
            if np.max(last_label) < 1:
                raise Exception("Error generating last label")
            else:
                print("Warning: low quality data for last label")
                break
    first_label.rio.write_crs(base_crs, inplace=True)
    last_label.rio.write_crs(base_crs, inplace=True)
    
    diff = xarray.where(first_label == 0, last_label, first_label)
    diff : xarray.Dataset = xarray.Dataset(
        {"water_change": (xarray.where(diff==2,1,0) - xarray.where(last_label==2,1,0)), 
         "start": first_label, 
         "finish": last_label}
    ).rio.write_crs(base_crs, inplace=True)
    diff = diff.rio.reproject(dst_crs=base_crs,resampling=Resampling.nearest,shape=(first_label.shape[-2]*10,first_label.shape[-1]*10))
    features : xarray.Dataset = rioxarray.open_rasterio(output_path).rio.reproject(dst_crs=base_crs,resampling=Resampling.bilinear)
    diff = diff.rio.clip_box(*features.rio.bounds(), crs=base_crs)
    diff = diff.rio.reproject(
        dst_crs=base_crs,
        shape=features.to_dataarray().shape[-2:] if isinstance(features, xarray.Dataset) else features.shape[-2:], # type: ignore
        resampling=Resampling.nearest,
    )
    #that's a kind of magic, the coords are the same but not 'exactly' the same, so we need to set them again
    diff.coords["x"] = features.coords["x"]
    diff.coords["y"] = features.coords["y"]
    diff.to_netcdf(os.path.join(folder_path, "labels.nc"))
    features.to_netcdf(os.path.join(folder_path, "features.nc"))
    diff.close()
    features.close()
    first_label.close()
    last_label.close()
    return first_label_path, last_label_path, output_path
    #if clear_temp_data:
    #    os.remove(first_label_path)
    #    os.remove(last_label_path)
    #    os.remove(output_path)


def generate_yearly_change_labels(event_location: Point, starting_event_date: date, ending_event_date: date, folder_path: str, features_path: str, clear_temp_data: bool = True, offset: float = 0.2):
    
    # Download the water masks for the starting and ending event dates, then save them to the folder_path
    
    first_labels_path = download_water_mask_yearly(
        event_location,
        starting_event_date.year,
        folder_path,
        offset=offset,
        water_type=WaterMaskType.PERMANENT,
        file_name=f"water_mask_start.tif",
    )
    last_labels_path = download_water_mask_yearly(
        event_location,
        ending_event_date.year,
        folder_path,
        offset=offset,
        water_type=WaterMaskType.PERMANENT,
        file_name=f"water_mask_end.tif",
    )
    print("First labels path: ", first_labels_path)
    print("Last labels path: ", last_labels_path)
    start = rioxarray.open_rasterio(first_labels_path)
    finish = rioxarray.open_rasterio(last_labels_path)
    
    # Calculate the difference between the starting and ending water masks and save it into a Dataset
    
    diff = xarray.DataArray(data=(start - finish),coords=start.coords,dims=start.dims)
    diff = xarray.where(diff != 0, xarray.where(diff > 0, 1, -1), 0)
    diff : xarray.Dataset = xarray.Dataset(
        {"water_change": diff, "start": start, "finish": finish}, coords=start.coords
    )
    
    # reproject both the features and the labels to the same CRS and shape
    # the double clip_box is necessary because the two rasters don't have the same bounds as the resolution is different
    
    features : xarray.Dataset = rioxarray.open_rasterio(features_path).rio.reproject(dst_crs=start.rio.crs,resampling=Resampling.bilinear)
    diff = diff.rio.clip_box(*features.rio.bounds(), crs=start.rio.crs)
    features = features.rio.clip_box(*diff.rio.bounds(), crs=start.rio.crs)
    diff = diff.rio.reproject(
        dst_crs=start.rio.crs,
        shape=features.to_dataarray().shape[-2:] if isinstance(features, xarray.Dataset) else features.shape[-2:], # type: ignore
        resampling=Resampling.nearest,
    )
    
    #that's a kind of magic, the coords are the same but not 'exactly' the same, so we need to set them again
    
    diff.coords["x"] = features.coords["x"]
    diff.coords["y"] = features.coords["y"]
    diff.to_netcdf(os.path.join(folder_path, "labels.nc"))
    diff.close()
    start.close()
    finish.close()
    features.to_netcdf(os.path.join(folder_path, "features.nc"))
    features.close()
    if clear_temp_data:
        print("Here I should've removed the temp data")