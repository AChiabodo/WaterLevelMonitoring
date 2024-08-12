import os
from datetime import date, timedelta
from shapely.geometry import Point
import xarray
import rioxarray
from rasterio.enums import Resampling
from water_mask import download_water_mask_yearly, WaterMaskType


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
    start.close()
    finish.close()
    
    #that's a kind of magic, the coords are the same but not 'exactly' the same, so we need to set them again
    
    diff.coords["x"] = features.coords["x"]
    diff.coords["y"] = features.coords["y"]
    diff.to_netcdf(os.path.join(folder_path, "labels.nc"))
    diff.close()
    features.to_netcdf(os.path.join(folder_path, "features.nc"))
    features.close()
    if clear_temp_data:
        os.remove(first_labels_path)
        os.remove(last_labels_path)
        print("Removed temporary files")
    return os.path.join(folder_path, "labels.nc"), os.path.join(folder_path, "features.nc")