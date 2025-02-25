import os
from datetime import date, timedelta
from pathlib import Path
from shapely.geometry import Point
import openeo
import pandas as pd
import logging

from utils import deg_to_dec, parse_args,BandsOptions, DRIVER_URL, PRODUCT_ID,  GREEN,  NIR,  SCL, MAX_CLOUD_COVER, DatasetOptions, check_value_in_all_csv
from labels import generate_yearly_change_labels
from tiles import multi_tiles
# from openeo.processes import ProcessBuilder as PGNode
from openeo.internal.graph_building import PGNode

logger = logging.getLogger(__name__)

"""
Download a patch of satellite data centered on a given location and date
:param event_location: the location of the event (longitude,latitude)
:param starting_event_date: the date of the event
:param base_output_path: the base path where to save the downloaded patch
:param name: the name of the patch
:param end_event_date: the end date of the patch (optional, default is the day after the starting date)
:param recurrence: the recurrence of the patch (optional, default is YEARLY) with valid values "YEARLY", "SEMIANNUAL" , "MONTHLY" or None for a single patch
:param time_step: the length in days of each observation (optional, default is 20)
:param max_cloud_cover: the maximum cloud cover allowed (optional, default is Constants.MAX_CLOUD_COVER)
:param offset: the offset from the center of the event location. The patch will be a square of size 2*offset (optional, default is 0.1)
"""
def download_temporal_patch(
    event_location: Point,
    starting_event_date: date,
    base_output_folder: Path,
    name: str,
    end_event_date: date | None = None,
    recurrence: str | None = "YEARLY", # valid values are "YEARLY", "SEMIANNUAL" , "MONTHLY" or None for a single patch
    time_step: int = 20, # how many days to include in each patch
    max_cloud_cover: int | None = MAX_CLOUD_COVER,
    offset: float = 0.1,
):
    if end_event_date is None:
        end_event_date = starting_event_date + timedelta(days=1)
    if (
        date.today() < starting_event_date
        or date.fromisoformat("2016-01-01") > starting_event_date
    ):
        print("Invalid starting date")
        return
    if recurrence == "SEMIANNUAL" and starting_event_date.month not in [5,6,7,11,12,1]:
        starting_event_date = starting_event_date.replace(month=5)
        print("Starting date changed to ", starting_event_date, " for tropical seasons recurrence")
    spatial_extent = {
        "west": event_location.x - offset,
        "east": event_location.x + offset,
        "north": event_location.y + offset,
        "south": event_location.y - offset,
        "crs": "EPSG:4326",
    }
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
            bands=BandsOptions.MSI.get_bands() + [SCL],
            max_cloud_cover=max_cloud_cover if max_cloud_cover is not None else MAX_CLOUD_COVER,
        )
        clouds = s2.band(SCL)
        clouds = (clouds == 3) | (clouds == 8) | (clouds == 9) | (clouds == 10)
        s2 = s2.mask(clouds)
        if recurrence == "MONTHLY" :
            s2 = s2.aggregate_temporal_period(period="month", reducer="max")
        elif recurrence == "YEARLY":
            s2 = s2.aggregate_temporal_period(period="year", reducer="max")
        elif recurrence == "SEMIANNUAL":
            s2 = s2.aggregate_temporal_period(period="tropical-season", reducer="max")
        else:
            s2 = s2.max_time()
        datacubes.append(s2)
        if recurrence == "YEARLY":
            event_date = event_date.replace(year=event_date.year + 1)
        elif recurrence == "MONTHLY":
            event_date = event_date.replace(day=1, month=(event_date.month % 12) + 1)
        elif recurrence == "SEMIANNUAL": #based on the tropical season
            if event_date.month <= 6:
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

    green = PGNode("array_element", arguments={"data": {"from_parameter": "data"}, "label": GREEN})
    nir = PGNode("array_element", arguments={"data": {"from_parameter": "data"}, "label": NIR})
    ndwi = PGNode("normalized_difference", arguments={"x": {"from_node": green}, "y": {"from_node": nir}})
    datacube_ndwi = datacube.reduce_dimension(dimension="bands", reducer=ndwi).add_dimension("bands", label="ndwi", type="bands")
    datacube = datacube.merge_cubes(datacube_ndwi)
    
    job = datacube.save_result(format="NetCDF").create_job(title="test_job")
    job.start_and_wait().get_results().download_file(output_path)
    return output_path


def instantiate_dataset(
    base_csv: str,
    base_output_path: str,
    datasets: list[DatasetOptions] = [],
    clear_data: bool = False,
    time_step: int = 30,
    offset: float = 0.1,
    max_cloud_cover: int = 15,
    full_dataset: bool = False, #if True downloads the full dataset ignoring the already downloaded patches
):
    df =  pd.read_csv(base_csv,header=0,usecols=["coordinates","name","category","downloaded","date","split","region"],index_col=["name"],sep=",")

    if not os.path.isdir(base_output_path):
        os.mkdir(base_output_path)
    target_csv = os.path.join(base_output_path,"dataset.csv")
    if not os.path.exists(target_csv):
        with open (target_csv, "w") as f:
            f.write("coordinates,name,category,date,downloaded,split,region\n")
    data_folder = os.path.join(base_output_path,"data")
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)
    
    for index, row in df[df["downloaded"] == 1].iterrows():
        
        target_df = pd.read_csv(target_csv,header=0,usecols=["coordinates","name","category","date","downloaded","split","region"],index_col=["name"],sep=",")
        current_date = date.fromisoformat(row["date"])
        folder_path = os.path.join(Path(data_folder), index.__str__())
        path_patch = labels_path = None
        
        # Skip if the patch is already downloaded and we're downloading the full dataset
        if not full_dataset and check_value_in_all_csv([target_csv] + [dataset.target_csv for dataset in datasets], index.__str__(),key_column="name"):
            print("Data already downloaded for ", index, " skipping")
            continue
        
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        
        try:
            # Download the temporal_patch from openeo if it's not already downloaded
            if not os.path.isfile(os.path.join(folder_path, "temporal_patch.nc")) :
                print("Downloading patch for ", index)
                path_patch = download_temporal_patch(
                    event_location=Point(deg_to_dec(row["coordinates"])), 
                    starting_event_date= current_date,
                    end_event_date= date.fromisoformat("2021-01-01"),
                    base_output_folder=folder_path,
                    time_step = time_step,
                    max_cloud_cover=max_cloud_cover,
                    offset=offset,
                    name=index.__str__(),
                    recurrence="YEARLY",
                )
            else:
                path_patch = os.path.join(folder_path, "temporal_patch.nc")
            print("Generating labels for ", index)
            #generate the labels and the transformed features for the patch
            labels_path, features_path = generate_yearly_change_labels(
                Point(deg_to_dec(row["coordinates"])),
                starting_event_date=current_date,
                ending_event_date=date.fromisoformat("2021-01-01"),
                folder_path=folder_path,
                features_path=path_patch,
                clear_temp_data=False,
                offset=offset,
            )
            if labels_path is not None and features_path is not None:
                if index not in target_df.index:
                    with open (target_csv, "ab") as f:
                        f.write(f"{row['coordinates']},{index.__str__()},{row['category']},{current_date},1,{row['split']},{row['region']}\n".encode('utf-8'))
                    print("New Patch downloaded for ", index , " at ", path_patch)
                else:
                    print("Patch already downloaded for ", index, ". Label Generated correctely")
            else:
                print("Error generating labels for ", index)
                continue
        except Exception as e:
            print("Error downloading patch for ", index, " error: ", e)
            continue
        
        tiling = multi_tiles(
            source_folder=folder_path,
            row=row,
            patch_name=index.__str__(),
            target_datasets=datasets,
            target_timestamp=[current_date,date.fromisoformat("2021-01-01")],
            nodata_allowed=0.15,
            save_change_mask=True,
            save_water_mask=True,
        )
        if tiling:
            print("Tiles generated for ", index)
        else:
            print("Error generating tiles for ", index)
            continue

        if clear_data and path_patch and labels_path:
            os.remove(features_path)
            os.remove(labels_path)

def main():
    args = parse_args()
    #TODO: Add the arguments,
    base_output_path = args.output
    old = [os.path.join(base_output_path,"NDWI_Annual_512"),
                            os.path.join(base_output_path,"NDWI_Annual_128"),
                            os.path.join(base_output_path,"NDWI_Annual_224")]
    target_folders=[os.path.join(base_output_path,"ALL_Annual_128"),
                    os.path.join(base_output_path,"ALL_Annual_384")],
    target_bands=[BandsOptions.ALL,BandsOptions.ALL],
    target_tile_size=[128,384],
    target_step_size=[128,384],

if __name__ == "__main__":
    instantiate_dataset(base_csv="baseDataset.csv",
                        base_output_path=os.path.join("datasets","base_Annual"),
                        offset=0.2,
                        clear_data=True,
                        full_dataset=False,
                        time_step=90,
                        max_cloud_cover=60,
                        datasets=[DatasetOptions(target_folder=os.path.join("datasets","base_Annual","ALL_Annual_128"),bands=BandsOptions.ALL,tile_size=128,step_size=128),
                                  DatasetOptions(target_folder=os.path.join("datasets","base_Annual","ALL_Annual_384"),bands=BandsOptions.ALL,tile_size=384,step_size=384),
                        ])