import os
from typing import List, Optional, Tuple
import math
import tempfile

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
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.applications import EfficientNetB0 as efn
import cloudpickle


def collection_validation(data: pd.DataFrame, collection_name: str):
    if collection_name not in data['asset_contract.name'].values:
        raise KeyError(f"{collection_name} is not conclude dataframe.")

    train_df = data[data['asset_contract.name'] != collection_name].reset_index(drop=True)
    test_df = data[data['asset_contract.name'] == collection_name].reset_index(drop=True)

    return train_df, test_df


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


def create_model(input_shape: Tuple[int], meta_shape: int,
                 output_shape: int, activation, loss,
                 learning_rate: float = 0.001,
                 pretrain: bool = False) -> models.Model:
    """
    The function for creating model.

    Parameters
    ----------
    input_shape : int
        Shape of input image data.
    meta_shape : int
        Shape of input meta data of image.
    output_shape : int
        Shape of model output.
    activation : function
        The activation function used hidden layers.
    loss : function
        The loss function of model.
    learning_rate : float
        The learning rate of model.
    pretrain : bool
        Flag that deterimine whether use pretrain model(default=False).

    Returns
    -------
    model : keras.models.Model
        Model instance.
    """
    if pretrain:
        weights = 'imagenet'
    else:
        weights = None

    inputs = layers.Input(shape=input_shape)
    efn_model = efn(include_top=False, input_shape=input_shape,
                    weights=weights)(inputs)
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
    model.compile(loss=loss,
                  optimizer=optim.SGD(learning_rate=learning_rate, momentum=0.9),
                  metrics=['mae', 'mse'])
    return model


class NFTModel(KerasRegressor):
    """
    Model class.
    This class is inherited KerasRegressor class of keras.
    """

    def __init__(self, model_func):
        """
        Constructor.

        Prameters
        ---------
        model_func : function
            The function for creating model.
        """
        super().__init__(build_fn=model_func)

    def __getstate__(self):
        result = {'sk_params': self.sk_params}
        with tempfile.TemporaryDirectory() as dir:
            if hasattr(self, 'model'):
                self.model.save(dir + '/output.h5', include_optimizer=False)
                with open(dir + '/output.h5', 'rb') as f:
                    result['model'] = f.read()
        return result

    def __setstate__(self, serialized):
        self.sk_params = serialized['sk_params']
        with tempfile.TemporaryDirectory() as dir:
            model_data = serialized.get('model')
            if model_data:
                with open(dir + '/input.h5', 'wb') as f:
                    f.write(model_data)
                self.model = tf.keras.models.load_model(dir + '/input.h5')

    def fit(self, train_gen, val_gen, epochs, batch_size, callbacks):
        """
        Training model.

        Parameters
        ----------
        train_gen : iterator
            The generator of train data.
        val_gen : iterator
            The generator of validation data.
        epochs : int
            Number of epochs for training model.
        batch_size : int
            Size of batch for training model.
        callbacks : list
            The list of callbacks.
            For example [EarlyStopping instance, ModelCheckpoint instance]
        """
        self.model = self.build_fn
        self.model.fit(train_gen, epochs=epochs, batch_size=batch_size,
                       validation_data=val_gen, callbacks=callbacks)

    def evaluate(self, test_X, test_y):
        """
        Evaluate model.

        Parameters
        ----------
        test_X : iterator
            The generator of test data.
        test_y : np.ndarray
            The array of targets of test data.
        """
        pred = self.model.predict(test_X)
        pred = np.where(pred < 0, 0, pred)
        rmse = np.sqrt(mean_squared_error(test_y, pred))
        mae = np.sqrt(mean_absolute_error(test_y, pred))

        print(f"RMSE Score: {rmse}")
        print(f"MAE Score: {mae}")

    def predict(self, img_path: str, collection_name: str, num_sales: int):
        """
        Predict data using trained model.

        Parameters
        ----------
        img_path : str
            The path of image data.
        collection_name : str
            Name of collection of the NFT.
        num_sales : int
            Number of times the NFT sold.
        """
        collection_dict = {
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
        meta_data = np.zeros(shape=(len(collection_dict)+1))
        if collection_name in self.collection_dict.keys():
            meta_data[self.collection_dict[collection_name]] = 1
        meta_data[-1] = num_sales
        meta_data = meta_data.reshape(1, -1)

        img = cv2.resize(cv2.imread(img_path)/256., (256, 256))
        img = img.reshape(1, 256, 256, 3)

        pred = self.model.predict([img, meta_data])
        return pred[0][0]


def train(path_list: np.ndarray, meta_data: np.ndarray,
          target: np.ndarray, loss):
    """
    The function for training model.

    Parameters
    ----------
    path_list : np.ndarray
        The path list of all image data.
    meta_data : np.ndarray
        The array of meta data of image.
    target : np.ndarray
        The array of targets data.
    loss : function
        The loss function of keras.
    """
    train_path, val_path, train_meta, val_meta, train_y, val_y =\
        train_test_split(path_list, meta_data, target, test_size=0.1, random_state=6174)

    train_gen = FullPathDataLoader(path_list=train_path,
                                   meta_data=train_meta, target=train_y,
                                   batch_size=16)
    val_gen = FullPathDataLoader(path_list=val_path,
                                 meta_data=val_meta, target=val_y,
                                 batch_size=1)
    model = NFTModel(
        create_model(input_shape=(256, 256, 3), meta_shape=len(meta_features),
                     output_shape=1, activation=activations.relu,
                     loss=loss, learning_rate=0.0001,
                     pretrain=True)
    )

    ES = callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                 restore_best_weights=True)

    print("starting training")
    print('*' + '-' * 30 + '*')

    model.fit(train_gen, val_gen, epochs=100, batch_size=16,
              callbacks=[ES])

    print("finished training")
    print('*' + '-' * 30 + '*' + '\n')

    val_gen = FullPathDataLoader(path_list=val_path,
                                 meta_data=val_meta, target=val_y,
                                 batch_size=1, shuffle=False, is_train=False)
    print("starting evaluate")
    print('*' + '-' * 30 + '*')

    model.evaluate(val_gen, val_y)

    print("finished evaluate")
    print('*' + '-' * 30 + '*' + '\n')

    return model


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
