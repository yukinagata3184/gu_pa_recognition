##
# @file record2ndarray.py
# @brief 学習/推論データとして使う音声を録音し配列として取得する目的で使用。
# @author yukinagata3184

import numpy as np
import pyaudio

def record2ndarray(CHUNK=1024, RATE=44100, RECORD_SECONDS=1, RECORD_START_THRESHOLD=1024,
                   CHANNELS=1, FORMAT=pyaudio.paInt16):
    """! 音声を録音し振幅をndarrayで返す関数
    @param CHUNK [int] 一回あたりに音声の振幅を取得する配列の要素数
    @param RATE [int] サンプリング周波数
    @param RECORD_SECONDS [int] 録音する秒数
    @param RECORD_START_THRESHOLD [int] 振幅がこの値を超えたとき録音を開始するしきい値
    @param CHANNELS [int] 1だとモノラル、2だとステレオ
    @param FORMAT [pyaudio] 録音するフォーマット
    @retval amplitude_array [np.ndarray] 録音した振幅を格納した配列
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

    NUM_INDEX = CHUNK * (RATE // CHUNK * RECORD_SECONDS)
    amplitude_array = np.zeros(NUM_INDEX, dtype="int16")

    while True:
        amplitude_bin_chunk = stream.read(CHUNK) # マイクからCHUNKずつ取得(バイナリ)
        amplitude_array_chunk = np.frombuffer(amplitude_bin_chunk, dtype="int16") # ndarrayに変換

        # 振幅がしきい値を上回ってからRECORD_SECONDS秒振幅を取得
        if np.max(np.abs(amplitude_array_chunk)) > RECORD_START_THRESHOLD:
            for i in range(0, RATE // CHUNK * RECORD_SECONDS):
                amplitude_bin_chunk = stream.read(CHUNK) # マイクからCHUNKずつ取得(バイナリ)
                amplitude_array[CHUNK*i:CHUNK*(i+1)] = np.frombuffer(amplitude_bin_chunk, dtype="int16")
            break
    return amplitude_array

if __name__ == "__main__":
    print("マイクに向かって「パ～～」と言ってください")
    amplitude_array = record2ndarray()
    np.set_printoptions(threshold=np.inf)
    print(amplitude_array)
    print("shape")
    print(amplitude_array.shape)
