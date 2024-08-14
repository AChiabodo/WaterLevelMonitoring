import os
from datetime import date, timedelta
from pathlib import Path
from shapely.geometry import Point
import openeo
import pandas as pd
import logging

from utils import deg_to_dec, parse_args,BandsOptions, DRIVER_URL, PRODUCT_ID,  GREEN,  NIR,  SCL, MAX_CLOUD_COVER
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
"""
def download_temporal_patch(
    event_location: Point,
    starting_event_date: date,
    base_output_folder: Path,
    name: str,
    end_event_date: date | None = None,
    recurrence: str | None = "YEARLY", # valid values are "YEARLY", "SEMIANNUAL" , "MONTHLY" or None for a single patch
    time_step: int = 20, # how many days to include in each patch
    max_cloud_cover: int | None = None,
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
    clear_data: bool = False,
    time_step: int = 45,
    offset: float = 0.1,
    tile_size: int = 256,
    step_size: int = 256,
    max_cloud_cover: int = 15,
    full_dataset: bool = False,
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
        if full_dataset and index in target_df.index:
            print("Data already downloaded for ", index, " skipping")
            continue
        
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        
        try:
            
            # Download the patch if it's not already downloaded
            if not os.path.isfile(os.path.join(folder_path, "temporal_patch.nc")) :
                print("Downloading patch for ", index)
                path_patch = download_temporal_patch(
                    event_location=Point(deg_to_dec(row["coordinates"])), 
                    starting_event_date= current_date,
                    end_event_date= date.fromisoformat("2021-01-01"),
                    base_output_folder=folder_path,
                    time_step = time_step,
                    max_cloud_cover=15,
                    offset=offset,
                    name=index.__str__(),
                )
            else:
                print("Patch already downloaded for ", index.__str__(), ". Generating labels ....")
                path_patch = os.path.join(folder_path, "temporal_patch.nc")
                
            
            if not os.path.isfile(os.path.join(folder_path, "features.nc")):
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
        
        multi_tiles(
            source_folder=folder_path,
            row=row,
            patch_name=index.__str__(),
            target_folders=[os.path.join(base_output_path,"NDWI_Annual_256"),
                            os.path.join(base_output_path,"MSI_Annual_256",),
                            os.path.join(base_output_path,"NDWI_Annual_512"),
                            os.path.join(base_output_path,"NDWI_Annual_128"),
                            os.path.join(base_output_path,"NDWI_Annual_224")],
            target_bands=[BandsOptions.NDWI,BandsOptions.MSI,BandsOptions.NDWI,BandsOptions.NDWI,BandsOptions.NDWI],
            target_tile_size=[256,256,512,128,224],
            target_step_size=[256,256,512,128,224],
            target_timestamp=[current_date,date.fromisoformat("2021-01-01")],
            nodata_allowed=0.1,
            save_change_mask=True,
            save_water_mask=True,
        )
        

        if clear_data and path_patch and labels_path:
            os.remove(features_path)
            os.remove(labels_path)

def main():
    args = parse_args()
    #TODO: Add the arguments

if __name__ == "__main__":
    instantiate_dataset(base_csv="baseDataset.csv",base_output_path=os.path.join("datasets","base_Annual"),offset=0.2, clear_data=True, full_dataset=True)
