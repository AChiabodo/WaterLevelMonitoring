from enum import Enum
import os
from datetime import date, timedelta
from pathlib import Path
from typing import List
from shapely.geometry import Point
import openeo
import rioxarray
import xarray

from utils import get_yearly_classification_file

def download_water_mask(event_location: Point, event_date: date,base_output_path: Path, name: str = "water_mask.tif", offset : float = 0.1):

    output_file = os.path.join(base_output_path,name)
    
    spatial_extent = {
        "west": event_location.x - offset, 
        "east": event_location.x + offset, 
        "north": event_location.y + offset, 
        "south": event_location.y - offset
    }
    temporal_extent = [
        str(event_date.year),
        str(event_date.year +1)
    ]
    print("Downloading water mask for ", name," location: ", event_location, " and date: ", event_date)
    print("spatial_extent: ", spatial_extent)
    # Connect to backend via basic authentication
    con = openeo.connect("https://earthengine.openeo.org/v1.0")
    con.authenticate_basic(username="group1", password="test123") # working only with testing account

    datacube = con.load_collection("JRC/GSW1_4/YearlyHistory",
                                spatial_extent=spatial_extent,
                                temporal_extent=temporal_extent,
                                bands=["waterClass"])

    # Send Job to backend
    job = datacube.save_result(format="GTIFF-THUMB",options={"epsgCode" : "EPSG:4326"}).create_job()

    # Wait for job to finish and download
    job.start_and_wait().get_results().download_file(output_file)
    return output_file

def download_water_mask_monthly(event_location: Point, event_date: date,base_output_path: Path, name: str = "water_mask.tif", offset : float = 0.2):

    output_file = os.path.join(base_output_path,name)

    spatial_extent = {
        "west": event_location.x - offset, 
        "east": event_location.x + offset, 
        "north": event_location.y + offset, 
        "south": event_location.y - offset
    }
    temporal_extent = [
        str(event_date - timedelta(days=60)),
        str(event_date + timedelta(days=30)),
    ]
    print("Downloading water mask for ", name," location: ", event_location, " and date: ", temporal_extent)
    print("spatial_extent: ", spatial_extent)
    # Connect to backend via basic authentication
    con = openeo.connect("https://earthengine.openeo.org/v1.0")
    con.authenticate_basic(username="group1", password="test123") # working only with testing account

    datacube = con.load_collection("JRC/GSW1_4/MonthlyHistory",
                                spatial_extent=spatial_extent,
                                temporal_extent=temporal_extent,
                                bands=["water"])
    datacube = datacube.max_time()
    # Send Job to backend
    job = datacube.save_result(format="GTIFF",options={"epsgCode" : "EPSG:4326"}).create_job()

    # Wait for job to finish and download
    job.start_and_wait().get_results().download_file(output_file)
    return output_file

class WaterMaskType(Enum):
    SEASONAL = 1
    PERMANENT = 2
    ALL = 3    

'''
Download the water mask (from Global Surface Water) for a specific year in a given location
:param event_location: The location of the event
:param event_year: The year of the event
:param base_output_path: The path where to save the downloaded water mask
:param offset: The "size" of the cube to download (default 0.2)
:param water_type: The type of water mask to download - Permantent Water, Seasonal Water or All Water
:return: The path of the downloaded water mask, containing 1 for water and 0 for non-water
'''
def download_water_mask_yearly(event_location: Point, event_year: date | int,base_output_path: Path, offset : float = 0.2,water_type: WaterMaskType = WaterMaskType.PERMANENT, file_name: str | None = None):
    if isinstance(event_year, date):
        event_year = event_year.year
    if file_name is None:
        output_file = os.path.join(base_output_path,"water_mask_year_{}.tif".format(event_year))
    else:
        output_file = os.path.join(base_output_path,file_name)
    
    print("Downloading water mask for year ", event_year, " location: ", event_location)
    file = get_yearly_classification_file(event_year, event_location.y, event_location.x)
    
    full_labels = rioxarray.open_rasterio(file).squeeze()
    spatial_extent = (event_location.x - offset, event_location.y - offset, event_location.x + offset, event_location.y + offset)
    labels = full_labels.rio.clip_box(*spatial_extent)
    crs = labels.rio.crs
    match (water_type):
        case WaterMaskType.PERMANENT:
            labels = xarray.DataArray(xarray.where(labels==3,1,0), dims=("y", "x"), coords={"y": labels.y, "x": labels.x}).rio.write_crs(crs, inplace=True)
        case WaterMaskType.SEASONAL:
            labels = xarray.DataArray(xarray.where(labels==2,1,0), dims=("y", "x"), coords={"y": labels.y, "x": labels.x}).rio.write_crs(crs, inplace=True)
        case _:
            raise Exception("Invalid water type")
    labels.rio.to_raster(output_file)
    labels.close()
    full_labels.close()
    return output_file

if __name__ == "__main__":
    res = download_water_mask_yearly(Point(32.5, 45.2), 2015, Path("."))
    res = rioxarray.open_rasterio(res)
    print(res.rio.bounds())
    print(res.rio.crs)
    print(res.max().values)