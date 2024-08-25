import os
from datetime import date, timedelta
from shapely.geometry import Point
import xarray
import rioxarray
from rasterio.enums import Resampling
from water_mask import download_water_mask_yearly, WaterMaskType
import gc

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
    start : xarray.DataArray = rioxarray.open_rasterio(first_labels_path)
    finish: xarray.DataArray = rioxarray.open_rasterio(last_labels_path )
    
    coords = start.coords
    dims = start.dims
    crs = start.rio.crs
    
    # Calculate the difference between the starting and ending water masks and save it into a Dataset
    # Doing (Start - Finish), if the resulting pixel is **positive**, it means that the pixel was water at the **start** and not in the **end** so it's a loss of water
    # Doing (Finish - Start), if the resulting pixel is **positive**, it means that the pixel was water in the **end** and not at the **start** so it's a gain of water
    diff = xarray.DataArray(data=(start - finish),coords=coords,dims=dims) # here we are doing (start - finish)
    
    # Here we are substituting the values of the diff array with 1 for positive values, -1 for negative values or 0 for "constant" values (no change)
    diff = xarray.where(diff != 0, xarray.where(diff > 0, 1, -1), 0)
    diff : xarray.Dataset = xarray.Dataset(
        {"water_change": diff, "start": start, "finish": finish}, coords=coords
    )
    
    # reproject both the features and the labels to the same CRS and shape
    # the double clip_box is necessary because the two rasters don't have the same bounds as the resolution is different
    
    #features : xarray.Dataset = rioxarray.open_rasterio(features_path).rio.reproject(dst_crs=start.rio.crs,resampling=Resampling.bilinear)
    #features : xarray.Dataset = xarray.open_dataset(features_path,decode_coords="all").rio.reproject(dst_crs=start.rio.crs,resampling=Resampling.bilinear)
    with xarray.open_dataset(features_path, decode_coords="all") as features:
        features = features.rio.reproject(dst_crs=crs,resampling=Resampling.bilinear)
        diff = diff.rio.clip_box(*features.rio.bounds(), crs=crs)
        features = features.rio.clip_box(*diff.rio.bounds(), crs=crs)
        diff = diff.rio.reproject(
            dst_crs=crs,
            shape=features.to_dataarray().shape[-2:] if isinstance(features, xarray.Dataset) else features.shape[-2:], # type: ignore
            resampling=Resampling.nearest,
        )
    
        #that's a kind of magic, the coords are the same but not 'exactly' the same, so we need to set them again
        
        diff.coords["x"] = features.coords["x"]
        diff.coords["y"] = features.coords["y"]
        diff.to_netcdf(os.path.join(folder_path, "labels.nc"))
        
        features.to_netcdf(os.path.join(folder_path, "features.nc"))
    
    diff = diff.close()
    start = start.close()
    finish = finish.close()

    if clear_temp_data:
        os.remove(first_labels_path)
        os.remove(last_labels_path)
        print("Removed temporary files")
    
    gc.collect()
    return os.path.join(folder_path, "labels.nc"), os.path.join(folder_path, "features.nc")