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
    init_angle = np.deg2rad(80)  # 初期角度（ラジアン）

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

        track = [results[0]]
        for peak in peaks:
            time = acce_all_df["app_timestamp"][peak]
            gyro_i = gyro_all_df["app_timestamp"].sub(time).abs().idxmin()

            x = step * np.cos(gyro_all_df["angle"][gyro_i] + init_angle) + track[-1][0]
            y = step * np.sin(gyro_all_df["angle"][gyro_i] + init_angle) + track[-1][1]

            track.append(Position(x, y, 0))

        # 推定結果を保存
        results.track = track

    results.plot_map()


if __name__ == "__main__":
    main()
