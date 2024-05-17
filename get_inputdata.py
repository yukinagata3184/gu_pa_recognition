##
# @file get_inputdata.py
# @brief ニューラルネットワークに入力するデータを取得する。
# @author yukinagata3184

import numpy as np
import time
from record2ndarray import record2ndarray
from preprocess_audio import *

def preprocess_formant(amplitude_array, offset=4096, frame_size=1024, sampling_freq=44100,
                       cepstrum_threshold=30, num_get_formant=2):
    """! 音声の振幅を格納した配列からフォルマント周波数を取得する
    @param amplitude_array [np.ndarray] 音声の振幅を格納した配列
    @param offset [int] 配列の何要素目から切り出すかを指定
    @param frame_size [int] フレーム切り出しの要素数を指定
    @param sampling_freq [int] 音声のサンプリング周波数
    @param cepstrum_threshold [int] 低ケフレンシ帯域と高ケフレンシ帯域の境界となる値
    @param num_get_formant [int] 第何フォルマントまで取得するか
    @retval formant_list [list] 第1フォルマントから順にnum_get_formantの数フォルマント周波数を格納したリスト
    """
    frame = frame_cutout(amplitude_array=amplitude_array, offset=offset, frame_size=frame_size)
    frame = hamming_window(frame=frame)
    frame, fs = fft(frame=frame, sampling_freq=sampling_freq, is_abs=True)
    frame = log(frame=frame)
    frame = ifft(frame=frame)
    frame = lowpass_lifter(frame=frame, cepstrum_threshold=cepstrum_threshold)
    frame, fs = fft(frame=frame, sampling_freq=sampling_freq, is_abs=False)
    formant_list = get_formant(frame=frame, sampling_freq=sampling_freq, num_get_formant=num_get_formant)
    return formant_list

def get_traindata(num_traindata=5, num_class=2):
    """! ニューラルネットワークに入力する学習データを取得する
    @param num_traindata [int] 1クラスあたりの学習データの数
    @param num_class [int] 分類する種類の数
    @retval traindata [np.ndarray] 学習データ
    @retval teacherdata [np.ndarray] 学習データの教師信号
    @retval normalization_max [float] 正規化の最大値
    """
    # 学習データを格納する配列を初期化
    traindata = np.zeros((num_traindata * num_class, 1), dtype="float32")
    # パーとグーを交互に学習させることで重みの偏りを防ぐ教師信号
    teacherdata = np.tile(np.array([[1, 0], [0, 1]], dtype="float32"), (num_traindata, 1))
    cnt = 1 # 何回目の学習データの取得かを表示するため使用

    print("ニューラルネットワークの学習に使うデータの取得を開始します")
    for i in range(num_traindata * num_class):
        if i % num_class == 0:
            print("「パ～～」と言ってください　" + str(cnt) + "/" + str(num_traindata))
        elif i % num_class == 1:
            print("「グ～～」と言ってください　" + str(cnt) + "/" + str(num_traindata))
            cnt += 1
        time.sleep(1) # 前の音声が入らないよう取得時間の調整
        amplitude_array = record2ndarray()
        traindata[i] = preprocess_formant(amplitude_array=amplitude_array)[0]
        print("val:" + str(traindata[i]))
        time.sleep(1) # 次の音声に入らないよう取得時間の調整
    normalization_max = np.max(traindata) # 正規化の最大値
    traindata = normalization(traindata, min_value=0, max_value=normalization_max)
    print("traindata:")
    print(traindata)
    return traindata, teacherdata, normalization_max

if __name__ == "__main__":
    get_traindata()