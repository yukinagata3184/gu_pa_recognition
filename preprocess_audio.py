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

def hamming_window(frame):
    """! フレーム切り出しした波形に周期性を持たせるためFFT前にハミング窓をかける
    @param frame [np.ndarray] フレーム切り出しした配列
    @param frame_size [int] フレームの要素数
    @return [np.ndarray] ハミング窓をかけた配列
    """
    return frame * np.hamming(len(frame))

def fft(frame, sampling_freq=44100, is_abs=False):
    """! FFTを行う
    @param frame [np.ndarray] FFTを行うフレームを格納した配列
    @param sampling_freq [int] 音声のサンプリング周波数
    @is_abs [bool] FFTの結果を絶対値にするか否か
    @retval frame [np.ndarray] FFTをかけた後の配列
    @retval fs [np.ndarray] x軸となる周波数を格納した配列
    """
    frame = np.fft.fft(frame)
    if is_abs:
        frame = np.abs(frame)
    fs = np.fft.fftfreq(len(frame), 1/sampling_freq)
    return frame, fs

def log(frame):
    """! 対数パワースペクトラムを求める
    @param frame [np.ndarray] 振幅スペクトラムを格納した配列
    @return [np.ndarray] 対数パワースペクトラムを格納した配列
    """
    return np.log(frame)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from record2ndarray import record2ndarray

    FRAME_SIZE = 1024
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))

    print("マイクに向かって「パ～～」と言ってください")
    amplitude_array = record2ndarray()

    frame = frame_cutout(amplitude_array=amplitude_array, frame_size=FRAME_SIZE)
    ax[0, 0].plot(list(range(FRAME_SIZE)), frame)
    ax[0, 0].set_title("frame cutout")
    ax[0, 0].set_xlabel("time")
    ax[0, 0].set_ylabel("amplitude")

    frame = hamming_window(frame=frame)
    ax[0, 1].plot(list(range(FRAME_SIZE)), frame)
    ax[0, 1].set_title("hamming window")
    ax[0, 1].set_xlabel("time")
    ax[0, 1].set_ylabel("amplitude")

    frame, fs = fft(frame=frame, is_abs=True)
    ax[0, 2].plot(fs[:int(FRAME_SIZE/2)], frame[:int(FRAME_SIZE/2)])
    ax[0, 2].set_title("amplitude spectrum")
    ax[0, 2].set_xlabel("freq[Hz]")
    ax[0, 2].set_ylabel("amplitude")

    frame = log(frame=frame)
    ax[0, 3].plot(fs[:int(FRAME_SIZE/2)], frame[:int(FRAME_SIZE/2)])
    ax[0, 3].set_title("log power spectrum")
    ax[0, 3].set_xlabel("freq[Hz]")
    ax[0, 3].set_ylabel("log power")

    plt.show()
