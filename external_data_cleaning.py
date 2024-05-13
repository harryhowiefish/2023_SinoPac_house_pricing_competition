import pandas as pd
import os
import pyproj
from tqdm import tqdm


def wgs84_to_twd97(pair: list) -> list:
    """
    Description: wgs84_to_twd97
    Input: [lat, lng] / [緯度,經度]
    Output: [easting,northing] / [x座標,y座標]

    """
    latitude, longitude = pair[0], pair[1]
    # Create a WGS84 to TWD97 projection transformer
    wgs84 = pyproj.CRS("EPSG:4326")  # WGS84 CRS
    twd97 = pyproj.CRS("EPSG:3826")  # TWD97 CRS

    transformer = pyproj.Transformer.from_crs(wgs84, twd97, always_xy=True)

    # Transform the coordinates
    easting, northing = transformer.transform(longitude, latitude)

    return [easting, northing]


def ext_cleaning(path: str):
    """
    Description:combine and simplify external data
    (only keeping location type) to a single csv
    Input: external_data folder path
    Output: None (csv file exported)

    """
    files = os.listdir(path)
    files = [file for file in files if file[-4:] == '.csv']
    external = {}
    for file in files:
        external[file[:-4]] = pd.read_csv(f'{path}/{file}')

    for name, table in external.items():
        result = []
        print(name)
        for pt in tqdm(table[['lat', 'lng']].to_numpy()):
            result.append(wgs84_to_twd97(pt))
        # table[['lng','lat']] = pd.DataFrame(result)

    external_simplified = pd.DataFrame()
    for name, table in external.items():
        temp = external[name][['lat', 'lng']]
        temp.insert(0, 'Name', [name]*len(temp))
        external_simplified = pd.concat([external_simplified, temp], axis=0)
    external_simplified = external_simplified.dropna()
    external_simplified.to_csv(
        f'{path}/external_transformed_simplified.csv', index=False)
