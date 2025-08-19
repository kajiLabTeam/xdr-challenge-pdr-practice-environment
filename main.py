import math
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

    window_acc_sec = 1.0  # 加速度の移動平均フィルタのウィンドウサイズ（秒）
    window_gyro_sec = 1.0  # 角速度の移動平均フィルタのウィンドウサイズ（秒）
    step = 0.4  # 歩幅（メートル）
    peak_distance_sec = 0.5  # ピーク検出の最小距離（秒）
    peak_height = 1.0  # ピーク検出の最小高さ
    #ピッチ角とロール角算出(初期姿勢)
    init_xangle = math.degrees(math.atan(dataprovider.acce_df["y"].at[0] / math.sqrt(dataprovider.acce_df["x"].at[0] ** 2 + dataprovider.acce_df["z"].at[0] ** 2)))
    init_yangle = math.degrees(math.atan(-dataprovider.acce_df["x"].at[0] / dataprovider.acce_df["z"].at[0]))
    
    #init_angle = np.deg2rad(80)  # 初期角度（ラジアン）
    #print(init_xangle)
    #print(init_yangle)
    
    #回転後の行列を格納
    rotated_acc = []
    
    for acce_df, gyro_df, acce_all_df, gyro_all_df in dataprovider:
        # サンプリング周波数の計算
        acce_fs = acce_all_df["app_timestamp"].count() / (
            acce_all_df["app_timestamp"].max() - acce_all_df["app_timestamp"].min()
        )
        gyro_fs = gyro_all_df["app_timestamp"].count() / (
            gyro_all_df["app_timestamp"].max() - gyro_all_df["app_timestamp"].min()
        )

        # ノルムの計算
        acce_all_df["norm"] = (
            acce_all_df["x"] ** 2 + acce_all_df["y"] ** 2 + acce_all_df["z"] ** 2
        ) ** (1 / 2)

        #回転行列
        r_x = np.array([
            [1, 0, 0],
            [0, np.cos(acce_all_df["x"]), -np.sin(acce_all_df["x"])],
            [0, np.sin(acce_all_df["x"]), np.cos(acce_all_df["x"])]
        ])
        r_y = np.array([
            [np.cos(acce_all_df["y"]), 0, np.sin(acce_all_df["y"])],
            [0, 1, 0],
            [-np.sin(acce_all_df["y"]), 0, np.cos(acce_all_df["y"])]
        ])
        r_z = np.array([
            [np.cos(acce_all_df["z"]), -np.sin(acce_all_df["z"]), 0],
            [np.sin(acce_all_df["z"]), np.cos(acce_all_df["z"]), 0],
            [0, 0, 1]
        ])
        
        #回転行列を使って世界座標に変換
        x = acce_all_df["x"]
        y = acce_all_df["y"]
        z = acce_all_df["z"]
        vec = np.array([x, y, z])
        vec = np.dot(r_x, vec)
        vec = np.dot(r_y, vec)
        vec = np.dot(r_z, vec)

        rotated_acc.append([
            acce_all_df["app_timestamp"],
            vec[0],
            vec[1],
            vec[2],
        ])

        #加速度、角速度それぞれで姿勢角を算出
        #2つの姿勢角から相補フィルタを作成し適用

        # 角度の計算
        gyro_all_df["angle"] = np.cumsum(gyro_all_df["x"]) / gyro_fs

        # 移動平均フィルタ
        window_acc_frame = int(window_acc_sec * acce_fs)
        window_gyro_frame = int(window_gyro_sec * gyro_fs)
        acce_all_df["low_norm"] = (
            acce_all_df["norm"].rolling(window=window_acc_frame).mean()
        )
        gyro_all_df["low_angle"] = (
            gyro_all_df["angle"].rolling(window=window_gyro_frame).mean()
        )

        # ピークの検出とプロット
        distance_frame = int(peak_distance_sec * acce_fs)
        peaks, _ = signal.find_peaks(
            acce_all_df["low_norm"],
            distance=distance_frame,
            height=peak_height,
        )

        track = [results.init_position]
        gyro_timestamps = gyro_all_df["app_timestamp"].values
        for peak in peaks:
            time = acce_all_df["app_timestamp"][peak]
            idx = np.searchsorted(gyro_timestamps, time)
            if idx == 0:
                gyro_i = gyro_all_df.index[0]
            elif idx == len(gyro_timestamps):
                gyro_i = gyro_all_df.index[-1]
            else:
                before = gyro_timestamps[idx - 1]
                after = gyro_timestamps[idx]
                if abs(time - before) <= abs(time - after):
                    gyro_i = gyro_all_df.index[idx - 1]
                else:
                    gyro_i = gyro_all_df.index[idx]

            x = step * np.cos(gyro_all_df["angle"][gyro_i] + init_xangle) + track[-1][0]
            y = step * np.sin(gyro_all_df["angle"][gyro_i] + init_yangle) + track[-1][1]

            track.append(Position(x, y, 0))

        # 推定結果を保存
        results.track = track

    #results.plot_map()

    for a in rotated_acc:
        print(a)


if __name__ == "__main__":
    main()
