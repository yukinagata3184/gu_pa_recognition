##
# @file neural_network.py
# @breif ニューラルネットワークの学習・推論・推論結果の後処理を行う。
# @author yukinagata3184

import numpy as np
import time
import tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from record2ndarray import record2ndarray
from preprocess_audio import normalization
from get_inputdata import preprocess_formant, get_traindata

def train(traindata, teacherdata, num_input=1, num_hidden_neuron=16, num_output_neuron=2, 
          hidden_activation="relu", optimizer="adam", epoch=500, seed=1):
    """! 3層(入力-中間-出力)のニューラルネットワークの学習を行う
    @param traindata [np.ndarray] 学習データ
    @param teacherdata [np.ndarray] 学習データの教師信号
    @param num_input [int] 入力層の数
    @param num_hidden_neuron [int] 中間層のニューロン数
    @param num_output_neuron [int] 出力層のニューロン数
    @param hidden_activation [str] 中間層の活性化関数の種類を指定
    @param optimizer [str] オプティマイザの種類を指定
    @param epoch [int] 学習を行うepoch数
    @param seed [int] ニューラルネットワークの初期値のseed値
    @retval model [keras.src.models.sequential.Sequential] 学習後のニューラルネットワークのオブジェクト
    """
    tensorflow.random.set_seed(seed)

    model = Sequential()
    model.add(Dense(num_hidden_neuron, activation=hidden_activation, input_shape=(num_input,))) # 中間層
    model.add(Dense(num_output_neuron, activation="softmax")) # 出力層

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['acc'])

    # 学習
    history = model.fit(traindata, teacherdata, epochs=epoch)

    return model

if __name__ == "__main__":
    traindata, teacherdata, normalization_max = get_traindata(num_traindata=5, num_class=2)
    model = train(traindata=traindata, teacherdata=teacherdata)
