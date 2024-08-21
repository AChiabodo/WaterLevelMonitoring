import os
import re
import requests
from enum import Enum
import argparse
import pandas as pd

DRIVER_URL = "https://openeo.dataspace.copernicus.eu/openeo/1.2"
# GEE_DRIVER_URL = "https://openeocloud.vito.be/openeo/1.0.0"
PRODUCT_ID = "SENTINEL2_L2A"
COASTAL = "B01"
BLUE = "B02"
GREEN = "B03"
RED = "B04"
RED_EDGE1 = "B05"
RED_EDGE2 = "B06"
RED_EDGE3 = "B07"
NIR = "B08"
SWIR16 = "B11"
SWIR22 = "B12"
SCL = "SCL"
MAX_CLOUD_COVER = 15

def deg_to_dec(coord):
    lat, lon = coord.split(' ')
    deg, minutes, seconds, direction =  re.split('[°\'"]', lat)
    lat = (float(deg) + float(minutes)/60 + float(seconds)/(60*60)) * (-1 if direction in ['W', 'S'] else 1)
    deg, minutes, seconds, direction =  re.split('[°\'"]', lon)
    lon = (float(deg) + float(minutes)/60 + float(seconds)/(60*60)) * (-1 if direction in ['W', 'S'] else 1)
    return lon, lat

def get_yearly_classification_file(year : int | str, latitudine : int | float, longitudine : int | float,saveDir : str = "yearlyClassification"):
    if isinstance(year, int):
        year = str(year)
    longitudine = int(( longitudine + 180 ) // 10 ) * 40000
    longitudine = str(longitudine).rjust(10, '0')
    latitudine = int(( 80 - latitudine ) // 10 ) * 40000
    latitudine = str(latitudine).rjust(10, '0')
    filename = "yearlyClassification" + year + "-" + latitudine + "-" + longitudine + ".tif"
    if os.path.exists(saveDir + "/" + filename):
        return saveDir + "/" + filename
    else:
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        link = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GSWE/YearlyClassification/VER5-0/tiles/yearlyClassification" + year + "/" + filename
        response = requests.get(link)
        if response.status_code == 200:
            with open(saveDir + "/" + filename, 'wb') as f:
                f.write(response.content)
            return saveDir + "/" + filename
        else:
            filename = "yearlyClassification" + year + "_" + latitudine + "-" + longitudine + ".tif"
            if os.path.exists(saveDir + "/" + filename):
                return saveDir + "/" + filename
            link = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GSWE/YearlyClassification/VER5-0/tiles/yearlyClassification" + year + "/" + filename
            response = requests.get(link)
            if response.status_code == 200:
                with open(saveDir + "/" + filename, 'wb') as f:
                    f.write(response.content)
                return saveDir + "/" + filename
            else:
                raise Exception("Error downloading yearly classification file")
            
class BandsOptions(Enum):
    NDWI = 1, # Normalized Difference Water Index
    RGB = 2, # Only B02, B03, B04
    MSI = 3, # B02, B03, B04, B08, 
    ALL = 4 # All bands
    NDVI = 5 # Normalized Difference Vegetation Index
    MNDWI = 6 # Modified Normalized Difference Water Index

    def get_bands(self, expand_indexes: bool = False):
        match self:
            case BandsOptions.NDWI:
                return [GREEN, NIR] if not expand_indexes else ["ndwi"]
            case BandsOptions.NDVI:
                return [NIR, RED] if not expand_indexes else ["ndvi"]
            case BandsOptions.MNDWI:
                return [GREEN, SWIR22] if not expand_indexes else ["mndwi"]
            case BandsOptions.RGB:
                return [BLUE, GREEN, RED]
            case BandsOptions.MSI:
                return [BLUE, GREEN, RED, NIR, SWIR16, SWIR22, RED_EDGE1, RED_EDGE2, RED_EDGE3]
            case BandsOptions.ALL:
                return [BLUE, GREEN, RED, NIR, SWIR16, SWIR22, RED_EDGE1, RED_EDGE2, RED_EDGE3, "ndwi"] # NDWI is not a band, but a derived index
    
    @classmethod
    def from_str(cls, value):
        if value in BandsOptions.__members__:
            return BandsOptions[value]
        else:
            return None

def str_to_enum(value):
    if value in BandsOptions.__members__:
        return BandsOptions[value]
    else:
        return None
            
class DatasetOptions():
    def __init__(self, target_folder: str, bands: BandsOptions, tile_size: int, step_size: int):
        self.target_csv = os.path.join(target_folder, "dataset.csv")
        self.target_folder = target_folder
        self.bands = bands
        self.tile_size = tile_size
        self.step_size = step_size
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        if not os.path.exists(self.target_csv):
            with open (self.target_csv, "w") as f:
                f.write("coordinates,name,category,date,downloaded,split,region,num_patches\n")
    
def check_value_in_all_csv(csv_files: list, value: str, key_column: str = "index"):
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, index_col=key_column)
        if value not in df.index:
            return False
    return True

def parse_args():
    parser = argparse.ArgumentParser(description="Download and process satellite data")
    parser.add_argument(
        "--mode",
        type=str,
        default="instantiate",
        help="Mode of operation: instantiate, download, tiles, labels",
    )
    parser.add_argument(
        "--base_csv", 
        type=str,
        default="datasets/baseDataset.csv" ,
        help="Base CSV file path")
    parser.add_argument(
        "--base_output_path", 
        type=str,
        default="datasets/RGB_Annual_256",
        help="Base output path")
    parser.add_argument(
        "--bands",
        type=str_to_enum,
        default=BandsOptions.NDWI,
        help="Selected bands to download")
    parser.add_argument(
        "--clear_temp_data",
        type=bool,
        default=False,
        help="Clear temporary data")
    parser.add_argument(
        "--clear_labels",
        type=bool,
        default=False,
        help="Clear labels data")
    parser.add_argument(
        "--tile_size",
        type=int,
        default=256,
        help="The size of a single tile")
    parser.add_argument(
        "--step_size",
        type=int,
        default=256,
        help="The step between tiles (if step < tile there will be some overlap)")
    parser.add_argument(
        "--time_step",
        type=int,
        default=45,
        help="The length in days of each observation")
    parser.add_argument(
        "--offset",
        type=float,
        default=0.2,
        help="The offset from the center of the event location. The patch will be a square of size 2*offset")