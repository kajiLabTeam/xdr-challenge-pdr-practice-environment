from pathlib import Path

import numpy as np
from scipy import signal
from src.results import Results
from src.data_provider import DataProvider
from src.type import Position


def main():
    src_dir = Path().resolve()
    acce_file = src_dir / "data" / "acce.csv"
    gyro_file = src_dir / "data" / "gyro.csv"
    map_file = src_dir / "map" / "miraikan_5.bmp"

    dataprovider = DataProvider(
        acce_file=acce_file,
        gyro_file=gyro_file,
        maxwait=0.5,
        offline=True,
    )
    results = Results(
        map_file=map_file,
        initial_position=Position(41.368, -10.047, 0),
    )

    #いい感じの値入れて調整
    # センサーの周波数(だいたい60)
    window_acc = 60
    window_gyro = 60

    step = 0.4
    peak_distance_sec = 0.5
    peak_heigh = 1.0


    #初期方向
    init_angle = np.deg2rad(75)

    for acce_df, gyro_df, acce_all_df, gyro_all_df in dataprovider:

        #サンプリング周波数計算
        acce_fs = acce_all_df["app_timestamp"].count() / (
            acce_all_df["app_timestamp"].max() - acce_all_df["app_timestamp"].min()
        )

        gyro_fs = gyro_all_df["app_timestamp"].count() / (
            gyro_all_df["app_timestamp"].max() - gyro_all_df["app_timestamp"].min()
        )

        #print(gyro_fs)
        #ノルム計算
        acce_all_df["norm"] = (
            acce_all_df["x"] ** 2 + acce_all_df["y"] ** 2 + acce_all_df["z"] ** 2
        ) ** (1 / 2)
        gyro_all_df["norm"] = (
            gyro_all_df["x"] ** 2 + gyro_all_df["y"] ** 2 + gyro_all_df["z"] ** 2
        ) ** (1 / 2)

        #積分
        #軸は持つスマホの向きで変わる
        gyro_all_df["angle"] = np.cumsum(gyro_all_df['x']) / gyro_fs

        #移動平均
        acce_all_df['low_norm'] = acce_all_df['norm'].rolling(window=window_acc).mean()
        gyro_all_df['low_angle'] = gyro_all_df['angle'].rolling(window=window_gyro).mean()

        #ピーク検出
        peaks, _= signal.find_peaks(acce_all_df["low_norm"], distance = peak_distance_sec * acce_fs, height = peak_heigh)

        #ピークごとの角速度取得
        track = [results[0]]
        print(peaks)
        for p in peaks:
            #時間の長さを合わせる
            time = acce_all_df['app_timestamp'][p]
            gyro_i = gyro_all_df["app_timestamp"].sub(time).abs().idxmin()

            x = step * np.cos(gyro_all_df['low_angle'][gyro_i] + init_angle) + track[-1][0]
            y = step * np.sin(gyro_all_df['low_angle'][gyro_i] + init_angle) + track[-1][1]

            track.append(Position(x, y, 0))

        #推定結果保存
        results.track = track

        

    results.plot_map()


if __name__ == "__main__":
    main()
