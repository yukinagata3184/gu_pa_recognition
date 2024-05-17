##
# @file preprocess_audio.py
# @brief 時間と振幅の2次元の音声データに対して前処理を行うことで次元を削減および特徴抽出し、
# @n ニューラルネットワークに入力できるデータを作成する。
# @author yukinagata3184

import numpy as np

def frame_cutout(amplitude_array, offset=4096, frame_size=1024):
    """! FFTを行うフレームを切り出す
    @param amplitude_array [np.ndarray] 音声の振幅を格納した配列
    @param offset [int] 配列の何要素目から切り出すかを指定
    @param frame_size [int] フレーム切り出しの要素数を指定
    @return [np.ndarray] フレーム切り出しした配列
    """
    return amplitude_array[offset:offset+frame_size]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from record2ndarray import record2ndarray

    FRAME_SIZE = 1024

    print("マイクに向かって「パ～～」と言ってください")
    amplitude_array = record2ndarray()
    frame = frame_cutout(amplitude_array=amplitude_array, frame_size=FRAME_SIZE)
    plt.plot(list(range(1024)), frame)
    plt.show()
