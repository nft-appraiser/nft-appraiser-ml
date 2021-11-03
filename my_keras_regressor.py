import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import tempfile

# クラス自体を変える場合はこっち
class MyKerasRegressor(KerasRegressor):
    def __getstate__(self):
        result = { 'sk_params': self.sk_params }
        with tempfile.TemporaryDirectory() as dir:
            if hasattr(self, 'model'): # 親Estimatorによるcloneなどで存在しないケースがある
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

# 既存のKerasRegressorをパッチする場合はこっち
def patch_keras_regressor():
    KerasRegressor.__getstate__ = _KerasRegressor__getstate__
    KerasRegressor.__setstate__ = _KerasRegressor__setstate__

def _KerasRegressor__getstate__(self):
    result = { 'sk_params': self.sk_params }
    with tempfile.TemporaryDirectory() as dir:
        if hasattr(self, 'model'): # 親Estimatorによるcloneなどで存在しないケースがある
            self.model.save(dir + '/output.h5', include_optimizer=False)
            with open(dir + '/output.h5', 'rb') as f:
                result['model'] = f.read()
    return result

def _KerasRegressor__setstate__(self, serialized):
    self.sk_params = serialized['sk_params']
    with tempfile.TemporaryDirectory() as dir:
        model_data = serialized.get('model')
        if model_data:
            with open(dir + '/input.h5', 'wb') as f:
                f.write(model_data)
            self.model = tf.keras.models.load_model(dir + '/input.h5')
