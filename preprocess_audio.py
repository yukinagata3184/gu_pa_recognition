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
    """! FFTを行い振幅スペクトラムを求める
    @param frame [np.ndarray] FFTを行うフレームを格納した配列
    @param sampling_freq [int] 音声のサンプリング周波数
    @param is_abs [bool] FFTの結果を絶対値にするか否か
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

def ifft(frame):
    """! 逆フーリエ変換を行いケプストラムを求める
    @param frame [np.ndarray] 対数パワースペクトラムを格納した配列
    @return [np.ndarray] ケプストラムを格納した配列
    """
    return np.fft.ifft(frame)

def lowpass_lifter(frame, cepstrum_threshold=30):
    """! ローパスリフタをかけ、高ケフレンシ帯域をカットする
    @param frame [np.ndarray] ケプストラムを格納した配列
    @param cepstrum_threshold [int] 低ケフレンシ帯域と高ケフレンシ帯域の境界となる値
    @return [np.ndarray] 高ケフレンシ帯域をカットしたケプストラムの配列
    """
    frame[cepstrum_threshold:len(frame)-cepstrum_threshold] = 0
    return frame

def get_formant(frame, sampling_freq=44100, num_get_formant=2):
    """! フォルマント(対数パワースペクトラムの包絡線の頂点)を取得する
    @param frame [np.ndarray] ケプストラムをフーリエ変換し対数パワースペクトラムの包絡線を抽出した配列
    @param sampling_freq [int] 音声のサンプリング周波数
    @param num_get_formant [int] 第何フォルマントまで取得するか
    @retval formant_list [list] 第1フォルマントから順にnum_get_formantの数フォルマント周波数を格納したリスト
    """
    # frameの1要素(目盛)あたり何Hzかを定義
    FS_ONE_SCALE = sampling_freq / len(frame)
    formant_list = []
    for axis in range(1, int(len(frame)/2)):
        # 対数パワースペクトラムの包絡線の頂点か否かを判定
        if frame[axis] > frame[axis-1] and frame[axis] > frame[axis+1]:
            formant_list.append(FS_ONE_SCALE * axis)
        # 指定の数フォルマント周波数を取得したら終了
        if len(formant_list) >= num_get_formant:
            break
    
    return formant_list

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
    log_power_spectrum = frame
    ax[0, 3].plot(fs[:int(FRAME_SIZE/2)], frame[:int(FRAME_SIZE/2)])
    ax[0, 3].set_title("log power spectrum")
    ax[0, 3].set_xlabel("freq[Hz]")
    ax[0, 3].set_ylabel("log power")

    frame = ifft(frame=frame)
    ax[1, 0].plot(list(range(int(FRAME_SIZE/2))), frame[:int(FRAME_SIZE/2)])
    ax[1, 0].set_title("quepstrum")
    ax[1, 0].set_xlabel("quefrency")
    ax[1, 0].set_ylabel("amplitude")

    frame = lowpass_lifter(frame=frame)
    ax[1, 1].plot(list(range(int(FRAME_SIZE/2))), frame[:int(FRAME_SIZE/2)])
    ax[1, 1].set_title("quepstrum")
    ax[1, 1].set_xlabel("quefrency")
    ax[1, 1].set_ylabel("amplitude")

    frame, fs = fft(frame=frame, is_abs=True)
    ax[1, 2].plot(fs[:int(FRAME_SIZE/2)], frame[:int(FRAME_SIZE/2)])
    ax[1, 2].set_title("log power spectrum")
    ax[1, 2].set_xlabel("freq[Hz]")
    ax[1, 2].set_ylabel("log power")

    ax[1, 3].plot(fs[:int(FRAME_SIZE/2)], log_power_spectrum[:int(FRAME_SIZE/2)])
    ax[1, 3].plot(fs[:int(FRAME_SIZE/2)], frame[:int(FRAME_SIZE/2)])
    ax[1, 3].set_title("log power spectrum")
    ax[1, 3].set_xlabel("freq[Hz]")
    ax[1, 3].set_ylabel("log power")

    formant_list = get_formant(frame=frame)
    print("F1: " + str(formant_list[0]))
    print("F2: " + str(formant_list[1]))

    plt.show()
