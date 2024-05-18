##
# @file neural_network.py
# @breif ニューラルネットワークの学習・推論・推論結果の後処理を行う。
# @author yukinagata3184

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.image import imread
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

def infer(model, normalization_max):
    """! 「グ～～」または「パ～～」の音声を取得し推論を行う
    @param model [keras.src.models.sequential.Sequential] 学習後のニューラルネットワークのオブジェクト
    @param normalization_max [float] 正規化の最大値
    @retval infer_result [np.ndarray] 推論結果をソフトマックス関数の形式で返す
    """
    # 推論するデータを格納する配列
    inferdata = np.zeros((1, 1), dtype="float32")
    print("「グ～～」または「パ～～」と言ってください")
    time.sleep(1) # 前の音声が入らないよう取得時間の調整
    amplitude_array = record2ndarray() # 推論するデータの取得
    inferdata[0, 0] = preprocess_formant(amplitude_array)[0]
    print("val:" + str(inferdata))
    time.sleep(1) # 次の音声に入らないよう取得時間の調整
    inferdata = normalization(inferdata, min_value=0, max_value=normalization_max) # 正規化
    infer_result = model.predict(inferdata) # 推論
    print("Softmax:" + str(infer_result))
    return infer_result

def judge_infer_result(infer_result):
    """! 推論結果の出力に応じて「グー」または「パー」の画像を表示する
    @param infer_result [np.ndarray] ニューラルネットワークからの推論結果を格納した配列
    """
    judge_index = np.argmax(infer_result)
    plt.close() # 前の画像を閉じる
    plt.gca().set_axis_off() # 画像から軸を消す
    if judge_index == 0:
        img = imread("img/janken_pa.png")
        plt.imshow(img)
        print("＊＊＊＊＊＊＊＊＊＊＊＊＊")
        print("＊パーの可能性が高いです＊")
        print("＊＊＊＊＊＊＊＊＊＊＊＊＊")
    else:
        img = imread("img/janken_gu.png")
        plt.imshow(img)
        print("＊＊＊＊＊＊＊＊＊＊＊＊＊")
        print("＊グーの可能性が高いです＊")
        print("＊＊＊＊＊＊＊＊＊＊＊＊＊")
    plt.pause(0.01)

if __name__ == "__main__":
    traindata, teacherdata, normalization_max = get_traindata(num_traindata=5, num_class=2)
    model = train(traindata=traindata, teacherdata=teacherdata)
    while True:
        infer_result = infer(model=model, normalization_max=normalization_max)
        judge_infer_result(infer_result)
        print("終了する際はCtrl+Cを押してください")
