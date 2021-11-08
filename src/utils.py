import os
import gc
import time
import imghdr
from io import BytesIO
from typing import List

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


def get_random_data(dir_name: str, num_loop: int) -> pd.DataFrame:
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

    get_features = ['id', 'asset.image_url', 'base_price', 'current_price', 'payment_token',
                    'quantity', 'asset.num_sales', 'asset.id', 'asset.token_id',
                    'asset.asset_contract.address', 'asset.asset_contract.asset_contract_type',
                    'asset.asset_contract.owner']

    df = pd.DataFrame(columns=get_features)
    img_id = 0
    url = "https://api.opensea.io/wyvern/v1/orders?bundled=false&include_bundled=false&include_invalid=false&limit=20&offset=0&order_by=created_date&order_direction=desc"
    for idx in tqdm(range(num_loop)):
        headers = {"Accept": "application/json"}
        params = {"limit": "50",
                  "offset": str(num_loop-idx)}

        response = requests.request("GET", url, headers=headers, params=params)

        data = response.json()
        orders_df = pd.json_normalize(data['orders'])

        for i in range(orders_df.shape[0]):
            try:
                img_url = orders_df.iloc[i]['asset.image_url']
                img_url.replace(" ", "")
                if is_image(img_url):
                    file_name = os.path.join(DATAPATH, f"{img_id}.png")
                    save_png(img_url, file_name)
                    df = df.append(orders_df.iloc[i][get_features])
                    img_id += 1
                elif is_svg(img_url):
                    file_name = os.path.join(DATAPATH, f"{img_id}.png")
                    save_svg(img_url, file_name)
                    df = df.append(orders_df.iloc[i][get_features])
                    img_id += 1
                else:
                    continue

            except:
                continue

        gc.collect()  # Just in case, free the memory so that the process does not stop
        time.sleep(60)

    df = df.reset_index(drop=True)
    df['image_id'] = df.index.values.astype(str)
    df['image_id'] = df['image_id'].apply(lambda x: x + '.png')
    return df


def get_collection_data(dir_name: str, target_collections: List[str]) -> pd.DataFrame:
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

    get_features = ['id', 'num_sales', 'image_url', 'asset_contract.name',
                    'owner.address', 'last_sale.quantity', 'last_sale.total_price']
    df = pd.DataFrame(columns=get_features)
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

                for i in range(assets_df.shape[0]):
                    img_url = assets_df.iloc[i]['image_url']
                    img_url.replace(" ", "")
                    if is_image(img_url):
                        file_name = os.path.join(DATAPATH, f"{img_id}.png")
                        save_png(img_url, file_name)
                        df = df.append(assets_df.iloc[i][get_features])
                        img_id += 1
                    elif is_svg(img_url):
                        file_name = os.path.join(DATAPATH, f"{img_id}.png")
                        save_svg(img_url, file_name)
                        df = df.append(assets_df.iloc[i][get_features])
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
