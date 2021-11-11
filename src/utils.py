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
                    is_test: bool = False) -> pd.DataFrame:
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
    img_id = 0
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

            response = requests.request("GET", url, params=params)

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
            gc.collect()  # Just in case, free the memory so that the process does not stop
            time.sleep(60)

    df = df.reset_index(drop=True)
    df['image_id'] = df.index.values.astype(str)
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
    headers = {"Accept": "application/json"}

    response = requests.request("GET", url, headers=headers)

    data = response.json()
    asset_df = pd.json_normalize(data)

    return asset_df
