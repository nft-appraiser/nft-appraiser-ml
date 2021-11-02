import os
from typing import List, Optional, Tuple
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import cv2
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optim
import tensorflow.keras.activations as activations
from tensorflow.keras.utils import Sequence
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.applications import EfficientNetB0 as efn
import cloudpickle


class FullPathDataLoader(Sequence):
    """
    Data loader that load images, meta data and targets.
    This class is inherited Sequence class of Keras.
    """

    def __init__(self, path_list: np.ndarray, meta_data: np.ndarray,
                 target: Optional[np.ndarray], batch_size: int, width: int = 256,
                 height: int = 256, resize: bool = True,
                 shuffle: bool = True, is_train: bool = True):
        """
        Constructor. This method determines class variables.

        Parameters
        ----------
        path_list : np.ndarray[str]
            The array of absolute paths of images.
        meta_data : np.ndarray[int]
            One-hot vector of collections.
        target : np.ndarray
            Array of target variavles.
        batch_size : int
            Batch size used when model training.
        width : int
            Width of resized image.
        height : int
            Height of resize image.
        resize : bool
            Flag determine whether to resize.
        shuffle : bool
            Flag determine whether to shuffle on epoch end.
        is_train : bool
            Determine whether this data loader will be used training model.
            if you won't this data loader, you have set 'is_train'=False.
        """
        self.path_list = path_list
        self.meta_data = meta_data
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.resize = resize
        self.shuffle = shuffle
        self.is_train = is_train
        self.length = math.ceil(len(self.path_list) / self.batch_size)

        if self.is_train:
            self.target = target

    def __len__(self):
        """
        Returns
        -------
        self.length : data length
        """
        return self.length

    def get_img(self, path_list: np.ndarray):
        """
        Load image data and resize image if 'resize'=True.

        Parameters
        ----------
        path_liist : np.ndarray
            The array of relative image paths from directory 'dir_name'.
            Size of this array is 'batch_size'.

        Returns
        -------
        img_list : np.ndarray
            The array of image data.
            Size of an image is (width, height, 3) if 'resize'=True.
        '"""
        img_list = []
        for path in path_list:
            img = cv2.imread(path)
            img = cv2.resize(img, (self.width, self.height))
            img = img / 255.
            img_list.append(img)

        img_list = np.array(img_list)
        return img_list

    def _shuffle(self):
        """
        Shuffle path_list, meta model.
        If 'is_train' is True, target is shuffled in association path_list.
        """
        idx = np.random.permutation(len(self.path_list))
        self.path_list = self.path_list[idx]
        self.meta_data = self.meta_data[idx]
        if self.is_train:
            self.target = self.target[idx]

    def __getitem__(self, idx):
        path_list = self.path_list[self.batch_size*idx:self.batch_size*(idx+1)]
        meta = self.meta_data[self.batch_size*idx:self.batch_size*(idx+1)]
        img_list = self.get_img(path_list)
        if self.is_train:
            target_list = self.target[self.batch_size*idx:self.batch_size*(idx+1)]

            return (img_list, meta), target_list
        else:
            return ((img_list, meta),)

    def on_epoch_end(self):
        if self.is_train:
            self._shuffle()


class DataLoader(Sequence):
    """
    Data loader that load images, meta data and targets.
    This class is inherited Sequence class of Keras.
    """

    def __init__(self, dir_name: str, path_list: np.ndarray, meta_data: np.ndarray,
                 target: Optional[np.ndarray], batch_size: int, width: int = 256,
                 height: int = 256, resize: bool = True,
                 shuffle: bool = True, is_train: bool = True):
        """
        Constructor. This method determines class variables.

        Parameters
        ----------
        dir_name : str
            Name of the directory that includes image data.
        path_list : np.ndarray[str]
            The array of relative paths of images from directory 'dir_name'.
        meta_data : np.ndarray[int]
            One-hot vector of collections.
        target : np.ndarray
            Array of target variavles.
        batch_size : int
            Batch size used when model training.
        width : int
            Width of resized image.
        height : int
            Height of resize image.
        resize : bool
            Flag determine whether to resize.
        shuffle : bool
            Flag determine whether to shuffle on epoch end.
        is_train : bool
            Determine whether this data loader will be used training model.
            if you won't this data loader, you have set 'is_train'=False.
        """
        self.dir_name = dir_name
        self.path_list = path_list
        self.meta_data = meta_data
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.resize = resize
        self.shuffle = shuffle
        self.is_train = is_train
        self.length = math.ceil(len(self.path_list) / self.batch_size)

        if self.is_train:
            self.target = target

    def __len__(self):
        """
        Returns
        -------
        self.length : data length
        """
        return self.length

    def get_img(self, path_list: np.ndarray):
        """
        Load image data and resize image if 'resize'=True.

        Parameters
        ----------
        path_liist : np.ndarray
            The array of relative image paths from directory 'dir_name'.
            Size of this array is 'batch_size'.

        Returns
        -------
        img_list : np.ndarray
            The array of image data.
            Size of an image is (width, height, 3) if 'resize'=True.
        '"""
        img_list = []
        if self.resize:
            for path in path_list:
                img = cv2.imread(os.path.join(self.dir_name, path))
                img = cv2.resize(img, (self.width, self.height))
                img = img / 255.
                img_list.append(img)

            img_list = np.array(img_list)
        else:
            for path in path_list:
                img = cv2.imread(os.path.join(self.dir_name, path))
                img = img / 255.
                img_list.append(img)

            img_list = np.array(img_list)
        return img_list

    def _shuffle(self):
        """
        Shuffle path_list, meta model.
        If 'is_train' is True, target is shuffled in association path_list.
        """
        idx = np.random.permutation(len(self.path_list))
        self.path_list = self.path_list[idx]
        self.meta_data = self.meta_data[idx]
        if self.is_train:
            self.target = self.target[idx]

    def __getitem__(self, idx):
        path_list = self.path_list[self.batch_size*idx:self.batch_size*(idx+1)]
        meta = self.meta_data[self.batch_size*idx:self.batch_size*(idx+1)]
        img_list = self.get_img(path_list)
        if self.is_train:
            target_list = self.target[self.batch_size*idx:self.batch_size*(idx+1)]

            return (img_list, meta), target_list
        else:
            return ((img_list, meta),)

    def on_epoch_end(self):
        if self.shuffle:
            self._shuffle()


class NFTModel:
    """
    Model Class.
    """

    def __init__(self, model_path: str):
        """
        Constructoer.

        Parameters
        ----------
        model_path : str
            The absolute path of model file.
        """
        self.model_path = model_path
        # 随時追加
        self.collection_dict = {
             'Axie': 0,
             'BoredApeYachtClub': 1,
             'CryptoPunks': 2,
             'CyberKongz': 3,
             'Doodles': 4,
             'GalaxyEggs': 5,
             'Jungle Freaks': 6,
             'KaijuKingz': 7,
             'Sneaky Vampire Syndicate': 8
        }

    def predict(self, img_path: str, collection_name: str, num_sales: int):
        """
        Predict Ethereum of new data. 

        Parameters
        ----------
        img_path : str
            The absolute path of image data.
        collection_name : str
            Collection name of NFT. This name will use as a feature if collection dict include this name.
        num_sales : int
            Number of times the NFT sold.
        """
        model = models.load_model(self.model_path)

        meta_data = np.zeros(shape=(len(self.collection_dict)+1))
        if collection_name in self.collection_dict.keys():
            meta_data[self.collection_dict[collection_name]] = 1
        meta_data[-1] = num_sales
        meta_data = meta_data.reshape(1, -1)

        img = cv2.resize(cv2.imread(img_path)/256., (256, 256))
        img = img.reshape(1, 256, 256, 3)

        pred = model.predict([img, meta_data])
        return pred[0][0]


def load_model(file_name: str):
    """
    Load the model file of pickle.

    Parameters
    ----------
    file_name : str
        The absolute path of the model file.

    Returns
    -------
    model : tf.keras.models.Model
        Trained model object.
    """
    with open(file_name, mode='rb') as f:
        model = cloudpickle.load(f)

    return model


def save_model(instance, file_name: str):
    """
    Save model as pickle file

    Parameters
    ----------
    instance : Class instance
        The class instance you want to save as pickle file.
    file_name : str
        The absolute path of file saved the instance.
    """
    with open(file_name, mode='wb') as f:
        cloudpickle.dump(instance, f)


def create_model(input_shape: Tuple[int], meta_shape: int,
                 output_shape: int, activation,
                 learning_rate: float = 0.001) -> models.Model:
    inputs = layers.Input(shape=input_shape)
    efn_model = efn(include_top=False, input_shape=input_shape,
                    weights=None)(inputs)
    ga = layers.GlobalAveragePooling2D()(efn_model)

    meta_inputs = layers.Input(shape=meta_shape)
    concate = layers.Concatenate()([ga, meta_inputs])
    dense1 = layers.Dense(units=128)(concate)
    bn1 = layers.BatchNormalization()(dense1)
    av1 = layers.Activation(activation)(bn1)
    dense2 = layers.Dense(units=64)(av1)
    bn2 = layers.BatchNormalization()(dense2)
    av2 = layers.Activation(activation)(bn2)
    outputs = layers.Dense(output_shape)(av2)

    model = models.Model(inputs=[inputs, meta_inputs], outputs=[outputs])
    model.compile(loss=losses.mean_absolute_error,
                  optimizer=optim.SGD(learning_rate=learning_rate, momentum=0.9),
                  metrics=['mae', 'mse'])
    return model

if __name__ == '__main__':
    pass

