import os
import gc
import time
import imghdr
from io import BytesIO
from typing import List, Optional

import requests
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm  # if you don't use IPython Kernel like jupyter, you should change "tqdm.notebook" to "tqdm"
from cairosvg import svg2png
from PIL import Image
import cv2

def get_opensea_api_key():
    return os.getenv('OPENSEA_API_KEY')

def is_image(url) -> bool:
    """
    Determine if it is an image of png or jpeg.

    Parameters
    ----------
    url : str
        Target url.

    Returns
    -------
    True or False: Return True if this url content is an image of png or jpeg else returns False.
    """
    img = requests.get(url).content
    img_type = imghdr.what(None, h=img)

    if img_type in ['png', 'jpeg']:
        return True
    else:
        return False


def is_svg(url) -> bool:
    """
    Determine if it is an image of svg.

    Parameters
    ----------
    url : str
        Target url.

    Returns
    -------
    True or False: Return True if this url content is an image of svg else returns False.
    """
    if url.endswith(".svg"):
        return True
    else:
        return False


def save_png(url, file_name) -> None:
    """
    Save an image of png or jpeg as a png file.

    Parameters
    ----------
    url : str
        Target url.
    file_name : str
        The file path of a saved png file.

    Returns
    -------
    None
    """
    img = requests.get(url).content
    img = Image.open(BytesIO(img)).convert("RGBA")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(file_name, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])


def save_svg(url, file_name) -> None:
    """
    Save an image of svg as an svg file. The content that is svg data of animation can't save.

    Parameters
    ----------
    url : str
        Target url.
    file_name : str
        The file path of a saved png file.

    Returns
    -------
    None
    """
    img = requests.get(url).content
    img = svg2png(bytestring=img)
    img = Image.open(BytesIO(img)).convert("RGBA")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(file_name, img)


def get_random_data(dir_name: str, num_loop: Optional[int] = None,
                    start_id: int = 0, is_test: bool = False) -> pd.DataFrame:
    """
    Get data of NFT to be had registered OpenSea by using OpenSea API.
    You can get a large amount of data randomly. If you want to change data to be acquired,
    you should view the reference of OpenSea API and change 'params' and 'get_features'.
    Also, I set the delay to reduce the server load by 1 process per 1 min.
    Please change according to your preference(within the bounds of common sense).
    If got data is other than png, jpeg, and svg(still image), the data can't save
    (but continue the process).

    Parameter
    ---------
    dir_name : str
        Directory path to save images.
    num_loop : int
        A number of loops. A number of getting data is 'num_loop' * 50
    is_test : bool
        Get small data(num_loop=5) regardless value of num_loop if you set "is_test=True".

    Returns
    -------
    df : pd.DataFrame
        The DataFrame consists of NFT data includes all image ids etc...

    See Also
    --------
    get_features : list of str
        There are hundreds of columns of original data, but the number of data to be
        acquired is limited. Please change according to your preference if column names
        that you need are not included.
    params : dict of requests parameters
        Like get_featrues, Please change according to your preference if you want to change
        getting data.
    """
    DATAPATH = dir_name

    df = pd.DataFrame()
    img_id = start_id
    url = "https://api.opensea.io/api/v1/assets"

    if is_test or num_loop is None:
        num_loop = 5
        print("This function execute on test mode(automatically changes to num_loop=5).")

    for idx in tqdm(range(num_loop)):
        try:
            params = {"limit": "50",
                      "order_by": "sale_date",
                      "order_direction": "desc",
                      "offset": str(50*idx)}

            response = requests.get(url, params=params)

            data = response.json()
            assets_df = pd.json_normalize(data['assets'])

            for feature in assets_df.columns.values:
                if feature not in df.columns.values:
                    df[feature] = None

            for feature in df.columns.values:
                if feature not in assets_df.columns.values:
                    assets_df[feature] = None

            for i in range(assets_df.shape[0]):
                img_url = assets_df.iloc[i]['image_url']
                img_url.replace(" ", "")
                if is_image(img_url):
                    file_name = os.path.join(DATAPATH, f"{img_id}.png")
                    save_png(img_url, file_name)
                    df = df.append(assets_df.iloc[i])
                    img_id += 1
                elif is_svg(img_url):
                    file_name = os.path.join(DATAPATH, f"{img_id}.png")
                    save_svg(img_url, file_name)
                    df = df.append(assets_df.iloc[i])
                    img_id += 1
                else:
                    continue

            gc.collect()  # Just in case, free the memory so that the process does not stop
            time.sleep(60)

        except:
            gc.collect()
            time.sleep(60)
            continue

    df = df.reset_index(drop=True)
    df['image_id'] = (df.index.values.astype(int)+start_id).astype(str)
    df['image_id'] = df['image_id'].apply(lambda x: x + '.png')
    return df


def get_collection_data(dir_name: str, target_collections: Optional[List[str]] = None,
                        is_test: bool = False) -> pd.DataFrame:
    """
    Get data of NFT to be had registered OpenSea by using OpenSea API.
    You can get a large amount of data you prefer collection. If you want to change data to be acquired,
    you should view the reference of OpenSea API and change 'params' and 'get_features'.
    Also, I set the delay to reduce the server load by 1 process per 1 min.
    Please change according to your preference(within the bounds of common sense).
    If got data is other than png, jpeg, and svg(still image), the data can't save
    (but continue the process).

    Parameter
    ---------
    dir_name : str
        Directory path to save images.
    target_collections : list of str
        The list of collection names you prefer.
        This variable can be set None, but you must set is_test=True.
    is_test : bool
        Get small data regardless values of target_collections if you set "is_test=True".

    Returns
    -------
    df : pd.DataFrame
        The DataFrame consists of NFT data includes all image ids etc...

    See Also
    --------
    get_features : list of str
        There are hundreds of columns of original data, but the number of data to be
        acquired is limited. Please change according to your preference if column names
        that you need are not included.
    params : dict of requests parameters
        Like get_featrues, Please change according to your preference if you want to change
        getting data.
    """
    DATAPATH = dir_name
    e_count = 0
    e_collection = []

    if is_test:
        print("This function execute on test mode.")
        print("Automatically set target_collections:\n['cryptopunks', 'boredapeyachtclub', 'doodles-official']")
        target_collections = ['cryptopunks', 'boredapeyachtclub', 'doodles-official']

    df = pd.DataFrame()
    img_id = 0
    url = "https://api.opensea.io/api/v1/assets"

    for collection in target_collections:
        for idx in tqdm(range(10), ascii=True, desc=collection):
            try:
                params = {
                    "offset": str(50*idx),
                    "order_by": "sale_date",
                    "order_direction": "desc",
                    "limit": "50",
                    "collection": collection
                }

                response = requests.get(url, params=params)

                data = response.json()
                assets_df = pd.json_normalize(data['assets'])

                for feature in assets_df.columns.values:
                    if feature not in df.columns.values:
                        df[feature] = None

                for feature in df.columns.values:
                    if feature not in assets_df.columns.values:
                        assets_df[feature] = None

                for i in range(assets_df.shape[0]):
                    img_url = assets_df.iloc[i]['image_url']
                    img_url.replace(" ", "")
                    if is_image(img_url):
                        file_name = os.path.join(DATAPATH, f"{img_id}.png")
                        save_png(img_url, file_name)
                        df = df.append(assets_df.iloc[i])
                        img_id += 1
                    elif is_svg(img_url):
                        file_name = os.path.join(DATAPATH, f"{img_id}.png")
                        save_svg(img_url, file_name)
                        df = df.append(assets_df.iloc[i])
                        img_id += 1
                    else:
                        continue

                gc.collect()  # Just in case, free the memory so that the process does not stop
                time.sleep(60)

            except:
                e_count += 1
                e_collection.append(collection)
                gc.collect()
                time.sleep(60)
                continue

    print(f"error count: {e_count}")
    print(f"error collection: {list(set(e_collection))}")
    df = df.reset_index(drop=True)
    df['image_id'] = df.index.values.astype(str)
    df['image_id'] = df['image_id'].apply(lambda x: x + '.png')
    return df


def get_data(asset_contract_address: str, token_id: str):
    """
    Get the asset data.

    Parameters
    ----------
    asset_contract_address : str
        The string of asset contract address.
    token_id : str
        The string of token id.

    Returns
    -------
    orders_df : pd.DataFrame
        The dataframe of asset data.
    """

    if type(token_id) != str:
        token_id = str(token_id)
    url = f"https://api.opensea.io/api/v1/asset/{asset_contract_address}/{token_id}/"

    response = requests.request("GET", url)

    data = response.json()
    asset_df = pd.json_normalize(data)

    return asset_df

def get_events_data(dir_name: str, num_loop: Optional[int] = None,
                    start_id: int = 0, is_test: bool = False) -> pd.DataFrame:
    """
    Get events data of NFT to be had registered OpenSea by using OpenSea API.
    You can get a large amount of data randomly. If you want to change data to be acquired,
    you should view the reference of OpenSea API and change 'params' and 'get_features'.
    Also, I set the delay to reduce the server load by 1 process per 1 min.
    Please change according to your preference(within the bounds of common sense).
    If got data is other than png, jpeg, and svg(still image), the data can't save
    (but continue the process).

    Parameter
    ---------
    dir_name : str
        Directory path to save images.
    num_loop : int
        A number of loops. A number of getting data is 'num_loop' * 50
    is_test : bool
        Get small data(num_loop=5) regardless value of num_loop if you set "is_test=True".

    Returns
    -------
    df : pd.DataFrame
        The DataFrame consists of NFT data includes all image ids etc...

    See Also
    --------
    get_features : list of str
        There are hundreds of columns of original data, but the number of data to be
        acquired is limited. Please change according to your preference if column names
        that you need are not included.
    params : dict of requests parameters
        Like get_featrues, Please change according to your preference if you want to change
        getting data.
    """
    DATAPATH = dir_name

    df = pd.DataFrame()
    img_id = start_id
    url = "https://api.opensea.io/api/v1/events?event_type=offer_entered&only_opensea=false&offset=0&limit=50"
    headers = {"Accept": "application/json",
               "X-API-KEY": get_opensea_api_key()}

    if is_test or num_loop is None:
        num_loop = 5
        print("This function execute on test mode(automatically changes to num_loop=5).")

    for idx in tqdm(range(num_loop)):
        try:
            params = {"limit": "50",
                      "offset": str(50*idx)}

            response = requests.get(url, params=params, headers=headers)

            data = response.json()
            assets_df = pd.json_normalize(data['asset_events'])

            for feature in assets_df.columns.values:
                if feature not in df.columns.values:
                    df[feature] = None

            for feature in df.columns.values:
                if feature not in assets_df.columns.values:
                    assets_df[feature] = None

            for i in range(assets_df.shape[0]):
                img_url = assets_df.iloc[i]['asset.image_url']
                img_url.replace(" ", "")
                if is_image(img_url):
                    file_name = os.path.join(DATAPATH, f"{img_id}.png")
                    save_png(img_url, file_name)
                    df = df.append(assets_df.iloc[i])
                    img_id += 1
                elif is_svg(img_url):
                    file_name = os.path.join(DATAPATH, f"{img_id}.png")
                    save_svg(img_url, file_name)
                    df = df.append(assets_df.iloc[i])
                    img_id += 1
                else:
                    continue

            gc.collect()  # Just in case, free the memory so that the process does not stop
            time.sleep(60)

        except:
            gc.collect()
            time.sleep(60)
            continue

    df = df.reset_index(drop=True)
    df['image_id'] = (df.index.values.astype(int)+start_id).astype(str)
    df['image_id'] = df['image_id'].apply(lambda x: x + '.png')
    return df

def concat_past_data(df: pd.DataFrame, num_past=10):
    """
    Get the NFTs events data. Concatenate successfule price data to df.

    Prameters
    ---------
    df : pd.DataFrame
        Dataframe of collection data.
    num_past : int
        Max number of the price data.
    """
    for i in range(num_past):
        df[f'past_price{i}'] = 0

    address_list = df['asset_contract.address'].values
    token_id_list = df['token_id'].values
    for idx, url_li in tqdm(enumerate(zip(address_list, token_id_list))):
        url = f"https://api.opensea.io/api/v1/events?asset_contract_address={url_li[0]}&token_id={url_li[1]}&only_opensea=false&offset=0&limit=50"

        headers = {"Accept": "application/json",
                   "X-API-KEY": get_opensea_api_key()}

        response = requests.request("GET", url, headers=headers)
        data = response.json()

        past_df = pd.json_normalize(data['asset_events'])
        price_list = past_df.query("event_type == 'successful'")['total_price'].values
        for i in range(min(num_past, len(price_list))):
            df.loc[idx, f'past_price{i}'] = price_list[i]

    return df

def get_past_data(df: pd.DataFrame, dir_name: str):
    """
    Get the NFTs events data as new dataframe. To use this function can get past data max 100.

    Prameters
    ---------
    df : pd.DataFrame
        Dataframe of the collection data.
    dir_name : str
        The name of directory you want to save data.
    """
    address_list = df['asset_contract.address'].values
    token_id_list = df['token_id'].values
    error_data = []
    for idx, url_li in tqdm(enumerate(zip(address_list, token_id_list))):
        try:
            url1 = f"https://api.opensea.io/api/v1/events?asset_contract_address={url_li[0]}&token_id={url_li[1]}&only_opensea=false&offset=0&limit=50"
            url2 = f"https://api.opensea.io/api/v1/events?asset_contract_address={url_li[0]}&token_id={url_li[1]}&only_opensea=false&offset=50&limit=50"

            headers = {"Accept": "application/json",
                       "X-API-KEY": get_opensea_api_key()}

            response1 = requests.request("GET", url1, headers=headers)
            data1 = response1.json()

            response2 = requests.request("GET", url2, headers=headers)
            data2 = response2.json()

            df1, df2 = pd.json_normalize(data1['asset_events']), pd.json_normalize(data2['asset_events'])

            df = pd.concat((df1, df2))
            file_name = f"{dir_name}/{asset_data.loc[idx, 'collection.name']}/{url_li[0]}_{url_li[1]}.csv"
            df.to_csv(file_name, index=False)
            gc.collect()

        except:
            gc.collect()
            continue

def get_successful_data(df: pd.DataFrame, dir_name: str):
    """
    Get the NFTs events data which successful transaction as new dataframe.

    Prameters
    ---------
    df : pd.DataFrame
        Dataframe of the collection data.
    dir_name : str
        The name of directory you want to save data.
    """
    address_list = df['asset_contract.address'].values
    token_id_list = df['token_id'].values
    for idx, url_li in tqdm(enumerate(zip(address_list, token_id_list))):
        try:
            url = f"https://api.opensea.io/api/v1/events?event_type=successful&asset_contract_address={url_li[0]}&token_id={url_li[1]}&only_opensea=false&offset=0&limit=50"

            headers = {"Accept": "application/json",
                       "X-API-KEY": get_opensea_api_key()}

            response = requests.request("GET", url, headers=headers)
            data = response.json()

            suc_df = pd.json_normalize(data['asset_events'])
            file_name = f"{dir_name}/{asset_data.loc[idx, 'collection.name']}/{url_li[0]}_{url_li[1]}_successful.csv"
            suc_df.to_csv(file_name, index=False)
            gc.collect()
        except:
            gc.collect()
            continue
