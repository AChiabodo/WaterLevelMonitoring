import os
from datetime import date, timedelta
from pathlib import Path
from typing import List
from shapely.geometry import Point
import openeo
import pandas as pd

from utils import deg_to_dec, parse_args,BandsOptions, DRIVER_URL, PRODUCT_ID, COASTAL, BLUE, GREEN, RED, NIR, SWIR16, SWIR22, SCL, MAX_CLOUD_COVER
from labels import generate_yearly_change_labels
from tiles import create_tiles
# from openeo.processes import ProcessBuilder as PGNode
from openeo.internal.graph_building import PGNode


"""
Download a patch of satellite data centered on a given location and date
:param event_location: the location of the event (longitude,latitude)
:param starting_event_date: the date of the event
:param base_output_path: the base path where to save the downloaded patch
"""
def download_temporal_patch(
    event_location: Point,
    starting_event_date: date,
    base_output_folder: Path,
    name: str,
    end_event_date: date | None = None,
    selected_bands: BandsOptions = BandsOptions.RGB, #valid values are
    recurrence: str | None = "YEARLY", # valid values are "YEARLY", "MONTHLY" or None for a single patch
    clear_temp_data: bool = True, # clear temporary data generated at runtime, preserve labels and features
    clear_labels: bool = True, # clear labels data
    clear_data: bool = True, # clear features data
    time_step: int = 20, # how many days to include in each patch
    max_cloud_cover: int | None = None,
    task : str = "change", # "change" or "single"
    offset: float = 0.1,
    reshape_size: int | None = None,
):
    if end_event_date is None:
        end_event_date = starting_event_date + timedelta(days=1)
    if (
        date.today() < starting_event_date
        or date.fromisoformat("2016-01-01") > starting_event_date
    ):
        print("Invalid starting date")
        return
    spatial_extent = {
        "west": event_location.x - offset,
        "east": event_location.x + offset,
        "north": event_location.y + offset,
        "south": event_location.y - offset,
        "crs": "EPSG:4326",
    }
    match selected_bands:
        case BandsOptions.NDWI:
            bands = [GREEN, NIR]
        case BandsOptions.RGB:
            bands = [BLUE, GREEN, RED]
        case BandsOptions.MSI:
            bands = [BLUE, GREEN, RED, NIR, SWIR22] #TODO
    output_path = os.path.join(base_output_folder, "temporal_patch.nc")
    con = openeo.connect(DRIVER_URL)
    con.authenticate_oidc()
    event_date = starting_event_date
    datacubes = []
    while event_date < end_event_date:  
        print(
            "Downloading patch for ",
            name,
            " location: ",
            event_location,
            " and date: ",
            event_date,
        )
        temporal_extent = [event_date, event_date + timedelta(days=time_step)]
        s2 = con.load_collection(
            collection_id=PRODUCT_ID,
            temporal_extent=temporal_extent,
            spatial_extent=spatial_extent,
            bands=bands + [SCL],
            max_cloud_cover=max_cloud_cover if max_cloud_cover is not None else MAX_CLOUD_COVER,
        )
        clouds = s2.band(SCL)
        clouds = (clouds == 3) | (clouds == 8) | (clouds == 9) | (clouds == 10)
        s2 = s2.mask(clouds)
        if recurrence == "MONTHLY":
            s2 = s2.aggregate_temporal_period(period="month", reducer="max")
        elif recurrence == "YEARLY":
            s2 = s2.aggregate_temporal_period(period="year", reducer="max")
        else:
            s2 = s2.max_time()
        datacubes.append(s2)
        if recurrence == "YEARLY":
            event_date = event_date.replace(year=event_date.year + 1)
        elif recurrence == "MONTHLY":
            event_date = event_date.replace(day=1, month=(event_date.month % 12) + 1)
        elif recurrence == "SEMIANNUAL":
            if event_date.month < 6:
                event_date = event_date.replace(month=event_date.month + 6)
            else:
                event_date = event_date.replace(month=event_date.month - 6,year=event_date.year+1)
        else:
            print("Invalid recurrence")
            return
    datacube: openeo.DataCube = None
    for cube in datacubes:
        if datacube is None:
            datacube = cube
        else:
            datacube = datacube.merge_cubes(cube, overlap_resolver="max")
    if selected_bands == BandsOptions.NDWI:
        green = PGNode("array_element", arguments={"data": {"from_parameter": "data"}, "label": GREEN})
        nir = PGNode("array_element", arguments={"data": {"from_parameter": "data"}, "label": NIR})
        ndwi = PGNode("normalized_difference", arguments={"x": {"from_node": green}, "y": {"from_node": nir}})
        datacube = datacube.reduce_dimension(dimension="bands", reducer=ndwi).add_dimension("bands", label="ndwi", type="bands")
    else:
        datacube = datacube.filter_bands(bands = bands)
    job = datacube.save_result(format="NetCDF").create_job(title="test_job")
    job.start_and_wait().get_results().download_file(output_path)
    return output_path
    tiles_number = create_tiles_temporal(folder_path, 256, 256, timestamp = [starting_event_date, event_date],reshape_size=reshape_size, nodata_allowed=0.1, save_change_mask=True)
    if clear_labels:
        os.remove(os.path.join(folder_path, "labels.nc"))
    if clear_data:
        os.remove(os.path.join(folder_path, "features.nc"))
    return tiles_number

def instantiate_dataset(
    base_csv: str,
    base_output_path: str,
    bands: BandsOptions | None = None,
    clear_temp_data: bool = True,
    clear_labels: bool = False,
    clear_data: bool = False,
    time_step: int = 45,
    offset: float = 0.1,
    tile_size: int = 256,
    step_size: int = 256,
):
    df =  pd.read_csv(base_csv,header=0,usecols=["coordinates","name","category","downloaded","date","split","region"],index_col=["name"],sep=",")

    if not os.path.isdir(base_output_path):
        os.mkdir(base_output_path)
    target_csv = os.path.join(base_output_path,"dataset.csv")
    if not os.path.exists(target_csv):
        with open (target_csv, "w") as f:
            f.write("coordinates,name,category,date,downloaded,split,region,num_patches\n")
    data_folder = os.path.join(base_output_path,"data")
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)
    for index, row in df[df["downloaded"] == 1].iterrows():
        target_df = pd.read_csv(target_csv,header=0,usecols=["coordinates","name","category","date","downloaded","split","num_patches","region"],index_col=["name"],sep=",")
        if index in target_df.index:
            continue
        current_date = date.fromisoformat(row["date"])
        valid_patch = False
        retry_count = 0
        folder_path = os.path.join(Path(data_folder), index.__str__())
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        while valid_patch is False and retry_count < 2:
            try:
                path_patch = download_temporal_patch(
                    event_location=Point(deg_to_dec(row["coordinates"])), 
                    starting_event_date= current_date,
                    end_event_date= date.fromisoformat("2021-01-01"),
                    base_output_folder= folder_path,
                    clear_temp_data = clear_temp_data,
                    clear_labels= clear_labels,
                    clear_data= clear_data,
                    time_step = time_step,
                    selected_bands = bands if bands is not None else BandsOptions.RGB,
                    max_cloud_cover=15,
                    offset=offset,
                    name=index.__str__(),
                )
                generate_yearly_change_labels(
                    Point(deg_to_dec(row["coordinates"])),
                    starting_event_date=current_date,
                    ending_event_date=date.fromisoformat("2021-01-01"),
                    folder_path=folder_path,
                    features_path=path_patch,
                    clear_temp_data=clear_temp_data,
                    offset=offset,
                )
                tiles_number = create_tiles(folder_path, tile_size, step_size, timestamp = [current_date, date.fromisoformat("2021-01-01")],reshape_size=None, nodata_allowed=0.1, save_change_mask=True, save_water_mask=True)
                if tiles_number is not None and tiles_number > 0:
                    valid_patch = True
                    with open (target_csv, "a") as f:
                        f.write(f"{row['coordinates']},{index.__str__()},{row['category']},{current_date},1,{row['split']},{row['region']},{tiles_number}\n")
                print("Patch downloaded for ", index , " at ", path_patch)
            except Exception as e:
                print("Error downloading patch for ", index, " error: ", e)
                retry_count += 1
                current_date = current_date.replace(day=1,month=(current_date.month%12)+1)
            # print("Labels generated for ", first_label_path, last_label_path, output_path)

def main():
    args = parse_args()

if __name__ == "__main__":
    instantiate_dataset(base_csv="datasets/baseDataset.csv",base_output_path=os.path.join("datasets","RGB_Annual_256"),bands=BandsOptions.RGB,offset=0.2)
