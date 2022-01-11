import os
import sys
from typing import List, Optional, Tuple
import math
import tempfile
import random

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

sys.path.append('../swintransformer')
from swintransformer import SwinTransformer


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

    def __init__(self, path_list: np.ndarray, target: Optional[np.ndarray] = None,
                 meta_data: Optional[np.ndarray] = None, batch_size: int = 16,
                 task: str = "B", width: int = 256, height: int = 256,
                 resize: bool = True, shuffle: bool = True, is_train: bool = True):
        """
        Constructor. This method determines class variables.

        Parameters
        ----------
        path_list : np.ndarray[str]
            The array of absolute paths of images.
        meta_data : np.ndarray[int]
            One-hot vector of collections.
        target : np.ndarray
            Array of target variables.
        batch_size : int
            Batch size used when model training.
        task : str
            Please determine this data loader will be used for task A or B(default=A).
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
        self.batch_size = batch_size
        self.task = task
        self.width = width
        self.height = height
        self.resize = resize
        self.shuffle = shuffle
        self.is_train = is_train
        self.length = math.ceil(len(self.path_list) / self.batch_size)

        if self.is_train:
            self.target = target
        if self.task == "A":
            self.meta_data = meta_data

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
            The array of relative image paths.
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
        if self.task == "A":
            self.meta_data = self.meta_data[idx]
        if self.is_train:
            self.target = self.target[idx]

    def __getitem__(self, idx):
        path_list = self.path_list[self.batch_size*idx:self.batch_size*(idx+1)]
        img_list = self.get_img(path_list)
        if self.is_train:
            target_list = self.target[self.batch_size*idx:self.batch_size*(idx+1)]
            if self.task == "A":
                meta = self.meta_data[self.batch_size*idx:self.batch_size*(idx+1)]
                return (img_list, meta), target_list
            else:
                return img_list, target_list
        else:
            if self.task == "A":
                meta = self.meta_data[self.batch_size*idx:self.batch_size*(idx+1)]
                return ((img_list, meta),)
            else:
                return img_list

    def on_epoch_end(self):
        if self.shuffle:
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


def create_model(input_shape: Tuple[int], output_shape: int,
                 activation, loss, meta_shape: Optional[int] = None,
                 task: str = "B", learning_rate: float = 0.001,
                 pretrain: bool = False) -> models.Model:
    """
    The function for creating model.

    Parameters
    ----------
    input_shape : int
        Shape of input image data.
    output_shape : int
        Shape of model output.
    activation : function
        The activation function used hidden layers.
    loss : function
        The loss function of model.
    meta_shape : int
        Shape of input meta data of image.
    task : str
        Please determine this model will be used for task A or B(default=A).
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
    base_model = SwinTransformer('swin_tiny_224', include_top=False, pretrained=True, use_tpu=False)(inputs)

    if task == "A":
        meta_inputs = layers.Input(shape=meta_shape)
        concate = layers.Concatenate()([base_model, meta_inputs])
        dense1 = layers.Dense(units=128)(concate)
        av1 = layers.Activation(activation)(dense1)
        dr1 = layers.Dropout(0.3)(av1)
        dense2 = layers.Dense(units=64)(dr1)
        av2 = layers.Activation(activation)(dense2)
        dr2 = layers.Dropout(0.3)(av2)
        outputs = layers.Dense(output_shape)(dr2)

        model = models.Model(inputs=[inputs, meta_inputs], outputs=[outputs])

    elif task == "B":
        dense1 = layers.Dense(units=128)(base_model)
        av1 = layers.Activation(activation)(dense1)
        dr1 = layers.Dropout(0.3)(av1)
        dense2 = layers.Dense(units=64)(dr1)
        av2 = layers.Activation(activation)(dense2)
        dr2 = layers.Dropout(0.3)(av2)
        outputs = layers.Dense(output_shape)(dr2)

        model = models.Model(inputs=[inputs], outputs=[outputs])

    else:
        raise Exception("Please set task is A or B.")

    model.compile(loss=loss,
                  optimizer=optim.Adam(learning_rate=learning_rate),
                  metrics=['mae', 'mse'])
    return model


class NFTModel(KerasRegressor):
    """
    Model class.
    This class is inherited KerasRegressor class of keras.
    """

    def __init__(self, model_func, input_shape, output_shape,
                 activation, loss, meta_shape, task, learning_rate, pretrain):
        """
        Constructor.

        Prameters
        ---------
        model_func : function
            The function for creating model.
        """
        self.model_func = model_func
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.loss = loss
        self.meta_shape = meta_shape
        self.task = task
        self.learning_rate = learning_rate
        self.pretrain = pretrain
        super().__init__(
            build_fn=model_func(input_shape, output_shape,
                                activation=activation, loss=loss,
                                meta_shape=meta_shape, task=task,
                                learning_rate=learning_rate, pretrain=pretrain)
        )
        self.model = self.build_fn

    def __getstate__(self):
        result = {'sk_params': self.sk_params,
                  'model_func': self.model_func,
                  'input_shape': self.input_shape,
                  'output_shape': self.output_shape,
                  'activation': self.activation,
                  'loss': self.loss,
                  'meta_shape': self.meta_shape,
                  'task': self.task,
                  'learning_rate': self.learning_rate,
                  'pretrain': self.pretrain}
        with tempfile.TemporaryDirectory() as dir:
            if hasattr(self, 'model'):
                self.model.save_weights(dir + '/output.h5')
                with open(dir + '/output.h5', 'rb') as f:
                    result['weights'] = f.read()
        return result

    def __setstate__(self, serialized):
        self.sk_params = serialized['sk_params']
        self.model_func = serialized['model_func']
        self.input_shape = serialized['input_shape']
        self.output_shape = serialized['output_shape']
        self.activation = serialized['activation']
        self.loss = serialized['loss']
        self.meta_shape = serialized['meta_shape']
        self.task = serialized['task']
        self.learning_rate = serialized['learning_rate']
        self.pretrain = serialized['pretrain']
        self.model = self.model_func(
                    self.input_shape, self.output_shape,
                    activation=self.activation, loss=self.loss,
                    meta_shape=self.meta_shape, task=self.task,
                    learning_rate=self.learning_rate, pretrain=self.pretrain
                )

        with tempfile.TemporaryDirectory() as dir:
            weight_data = serialized.get('weights')
            if weight_data:
                with open(dir + '/input.h5', 'wb') as f:
                    f.write(weight_data)
                self.model.load_weights(dir + '/input.h5')

    def fit(self, train_gen, val_gen, epochs, batch_size, callbacks=None):
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

    def predict(self, img: np.ndarray, collection_name: str, num_sales: int):
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
        if self.task == "A":
            collections = ['CryptoPunks',
                           'Bored Ape Yacht Club',
                           'Edifice by Ben Kovach',
                           'Mutant Ape Yacht Club',
                           'The Sandbox',
                           'Divine Anarchy',
                           'Cosmic Labs',
                           'Parallel Alpha',
                           'Art Wars | AW',
                           'Neo Tokyo Identities',
                           'Neo Tokyo Part 2 Vault Cards',
                           'Cool Cats NFT',
                           'CrypToadz by GREMPLIN',
                           'BearXLabs',
                           'Desperate ApeWives',
                           'Decentraland',
                           'Neo Tokyo Part 3 Item Caches',
                           'Doodles',
                           'The Doge Pound',
                           'Playboy Rabbitars Official',
                           'THE SHIBOSHIS',
                           'THE REAL GOAT SOCIETY',
                           'Sipherian Flash',
                           'Party Ape | Billionaire Club',
                           'Treeverse',
                           'Angry Apes United',
                           'CyberKongz',
                           'Emblem Vault [Ethereum]',
                           'Fat Ape Club',
                           'VeeFriends',
                           'JUNGLE FREAKS BY TROSLEY',
                           'Meebits',
                           'Furballs.com Official',
                           'Kaiju Kingz',
                           'Bears Deluxe',
                           'PUNKS Comic',
                           'Hor1zon Troopers',
                           'Lazy Lions',
                           'LOSTPOETS',
                           'Chain Runners',
                           'Chromie Squiggle by Snowfro',
                           'MekaVerse',
                           'Vox Collectibles',
                           'MutantCats',
                           'World of Women',
                           'SuperFarm Genesis Series',
                           'Eponym by ART AI',]
            collection_dict = {
                 collections[i]: i for i in range(len(collections))
            }
            meta_data = np.zeros(shape=(len(collection_dict)+1))
            if collection_name in collection_dict.keys():
                meta_data[collection_dict[collection_name]] = 1
            meta_data[-1] = num_sales
            meta_data = meta_data.reshape(1, -1)

            img = cv2.resize(img/255., (224, 224))
            img = img.reshape(1, 224, 224, 3)

            pred = self.model.predict([img, meta_data])
        elif self.task == "B":
            img = cv2.resize(img/255., (224, 224))
            img = img.reshape(1, 224, 224, 3)

            pred = self.model.predict(img)

        return pred[0][0]


def train(path_list: np.ndarray, target: np.ndarray, loss,
          meta_data: Optional[np.ndarray] = None, task: str = "B"):
    """
    The function for training model.

    Parameters
    ----------
    path_list : np.ndarray
        The path list of all image data.
    target : np.ndarray
        The array of targets data.
    loss : function
        The loss function of keras.
    meta_data : np.ndarray
        The array of meta data of image.
    task : str
        Please determine you train model for task A or B(default=A).
    """
    if task == "A":
        train_path, val_path, train_meta, val_meta, train_y, val_y =\
            train_test_split(path_list, meta_data, target, test_size=0.1, random_state=6174)
        train_gen = FullPathDataLoader(path_list=train_path, target=train_y,
                                       meta_data=train_meta, batch_size=16,
                                       width=224, height=224, task=task)
        val_gen = FullPathDataLoader(path_list=val_path, target=train_y,
                                     meta_data=val_meta, batch_size=16,
                                     width=224, height=224, task=task)
    elif task == "B":
        train_path, val_path, train_y, val_y =\
            train_test_split(path_list, target, test_size=0.1, random_state=6174)
        train_gen = FullPathDataLoader(path_list=train_path, target=train_y,
                                       width=224, height=224, batch_size=16, task=task)
        val_gen = FullPathDataLoader(path_list=val_path, target=val_y,
                                     width=224, height=224, batch_size=16, task=task)
    else:
        raise Exception("Please set task is A or B")

    if meta_data:
        meta_shape = meta_data.shape[1]
    else:
        meta_shape = None

    set_seed()
    model = NFTModel(
        model_func=create_model, input_shape=(224, 224, 3),
        output_shape=1,activation=activations.relu, loss=loss,
        meta_shape=meta_shape, task=task,
        learning_rate=0.00001, pretrain=True
    )

    ES = callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                 restore_best_weights=True)

    print("starting training")
    print('*' + '-' * 30 + '*')

    model.fit(train_gen, val_gen, epochs=100, batch_size=16,
              callbacks=[ES])

    print("finished training")
    print('*' + '-' * 30 + '*' + '\n')

    if task == "A":
        val_gen = FullPathDataLoader(path_list=val_path, target=val_y,
                                     meta_data=val_meta, batch_size=1, task=task,
                                     width=224, height=224, shuffle=False, is_train=False)
    else:
        val_gen = FullPathDataLoader(path_list=val_path, target=val_y,
                                     batch_size=1, task=task,
                                     width=224, height=224, shuffle=False, is_train=False)
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
        
def set_seed(random_state=6174):
    tf.random.set_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)
    os.environ['PYTHONHASHSEED'] = str(random_state)
