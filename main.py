from pathlib import Path
import numpy as np
from scipy import signal
from src.results import Results
from src.data_provider import DataProvider
from src.type import Position
import pandas as pd
import numba as nb


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
        # 加速度と角速度を合体して管理
        all_df = pd.merge_asof(
            acce_all_df.rename(
                columns={
                    "x": "acce_x",
                    "y": "acce_y",
                    "z": "acce_z",
                    "accuracy": "acce_accuracy",
                }
            ),
            gyro_all_df.rename(
                columns={
                    "x": "gyro_x",
                    "y": "gyro_y",
                    "z": "gyro_z",
                    "accuracy": "gyro_accuracy",
                }
            ),
            on="app_timestamp",
            direction="nearest",
        )

        # 重力方向を計算(ノルム)
        all_df["norm"] = (
            all_df["acce_x"] ** 2 + all_df["acce_y"] ** 2 + all_df["acce_z"] ** 2
        ) ** (1 / 2)

        all_df["ori_x"] = np.arccos(all_df["acce_x"] / all_df["norm"])
        all_df["ori_y"] = np.arccos(all_df["acce_y"] / all_df["norm"])
        all_df["ori_z"] = np.arccos(all_df["acce_z"] / all_df["norm"])

        all_df[["ori_x", "ori_y", "ori_z"]] = np.vstack(
            update_ori(
                all_df["norm"].to_numpy(),
                all_df["ori_x"].to_numpy(),
                all_df["ori_y"].to_numpy(),
                all_df["ori_z"].to_numpy(),
                all_df["gyro_x"].to_numpy(),
                all_df["gyro_y"].to_numpy(),
                all_df["gyro_z"].to_numpy(),
            )
        ).T

        # 回転行列
        all_df["r_x"] = all_df.apply(
            lambda row: np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(row["ori_x"]), -np.sin(row["ori_x"])],
                    [0, np.sin(row["ori_x"]), np.cos(row["ori_x"])],
                ]
            ),
            axis=1,
        )
        all_df["r_y"] = all_df.apply(
            lambda row: np.array(
                [
                    [np.cos(row["ori_y"]), 0, np.sin(row["ori_y"])],
                    [0, 1, 0],
                    [-np.sin(row["ori_y"]), 0, np.cos(row["ori_y"])],
                ]
            ),
            axis=1,
        )
        all_df["r_z"] = all_df.apply(
            lambda row: np.array(
                [
                    [np.cos(row["ori_z"]), -np.sin(row["ori_z"]), 0],
                    [np.sin(row["ori_z"]), np.cos(row["ori_z"]), 0],
                    [0, 0, 1],
                ]
            ),
            axis=1,
        )

        # 世界座標系に変換
        all_df[["global_acce_x", "global_acce_y", "global_acce_z"]] = all_df.apply(
            lambda row: pd.Series(
                row["r_z"]
                @ row["r_y"]
                @ row["r_x"]
                @ np.array([row["acce_x"], row["acce_y"], row["acce_z"]])
            ),
            axis=1,
        )
        all_df[["global_gyro_x", "global_gyro_y", "global_gyro_z"]] = all_df.apply(
            lambda row: pd.Series(
                row["r_z"]
                @ row["r_y"]
                @ row["r_x"]
                @ np.array([row["gyro_x"], row["gyro_y"], row["gyro_z"]])
            ),
            axis=1,
        )

        # サンプリング周波数の計算
        fs = all_df["app_timestamp"].count() / (
            all_df["app_timestamp"].max() - all_df["app_timestamp"].min()
        )

        # 角度の計算
        all_df["angle"] = np.cumsum(all_df["global_gyro_x"]) / fs

        # 移動平均フィルタ
        window_acc_frame = int(window_acc_sec * fs)
        window_gyro_frame = int(window_gyro_sec * fs)
        all_df["low_norm"] = all_df["norm"].rolling(window=window_acc_frame).mean()
        all_df["low_angle"] = all_df["angle"].rolling(window=window_gyro_frame).mean()

        # ピークの検出とプロット
        distance_frame = int(peak_distance_sec * fs)
        peaks, _ = signal.find_peaks(
            all_df["low_norm"],
            distance=distance_frame,
            height=peak_height,
        )

        track = [results.init_position]
        gyro_timestamps = all_df["app_timestamp"].values
        for peak in peaks:
            time = all_df["app_timestamp"][peak]
            idx = np.searchsorted(gyro_timestamps, time)
            if idx == 0:
                gyro_i = all_df.index[0]
            elif idx == len(gyro_timestamps):
                gyro_i = all_df.index[-1]
            else:
                before = gyro_timestamps[idx - 1]
                after = gyro_timestamps[idx]
                if abs(time - before) <= abs(time - after):
                    gyro_i = all_df.index[idx - 1]
                else:
                    gyro_i = all_df.index[idx]

            x = step * np.cos(all_df["angle"][gyro_i] + init_angle) + track[-1][0]
            y = step * np.sin(all_df["angle"][gyro_i] + init_angle) + track[-1][1]

            track.append(Position(x, y, 0))

        # 推定結果を保存
        results.track = track

    results.plot_map()


# 歩行時と停止時の切り替え
@nb.njit
def update_ori(norm, ori_x, ori_y, ori_z, gx, gy, gz):
    for i in range(1, len(norm)):
        # 歩行中の場合
        if norm[i] >= 9.9:
            ori_x[i] = ori_x[i - 1] + gx[i]
            ori_y[i] = ori_y[i - 1] + gy[i]
            ori_z[i] = ori_z[i - 1] + gz[i]
    return ori_x, ori_y, ori_z


if __name__ == "__main__":
    main()
