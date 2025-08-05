from pathlib import Path
import numpy as np
import pandas as pd
from src.results import Results
from src.data_provider import DataProvider
from src.type import Position
from scipy import signal


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
        initial_position=Position(41.368, -10.047, 0.935),
    )

    window_acc = 60
    window_gyro = 60
    peak_distance = 30
    peak_height = 1.0
    step = 0.4
    init_angle = np.deg2rad(80)

    for acce_df, gyro_df, acce_all_df, gyro_all_df in dataprovider:
        gyro_fs = gyro_all_df["app_timestamp"].count() / (
            gyro_all_df["app_timestamp"].max() - gyro_all_df["app_timestamp"].min()
        )

        # 位置推定
        acce_all_df["norm"] = acce_all_df[["x", "y", "z"]].apply(norm, axis=1)
        gyro_all_df["norm"] = gyro_all_df[["x", "y", "z"]].apply(norm, axis=1)

        gyro_all_df["angle"] = gyro_all_df["x"].cumsum() / gyro_fs

        acce_all_df["low_norm"] = acce_all_df["norm"].rolling(window=window_acc).mean()
        gyro_all_df["low_x"] = gyro_all_df["x"].rolling(window=window_gyro).mean()
        gyro_all_df["low_angle"] = (
            gyro_all_df["angle"].rolling(window=window_gyro, center=True).mean()
        )

        peaks, _ = signal.find_peaks(
            acce_all_df["low_norm"], distance=peak_distance, height=peak_height
        )
        track = [results[0]]
        for p in peaks:
            time = acce_all_df["app_timestamp"][p]
            gyro_i = gyro_all_df["app_timestamp"].sub(time).abs().idxmin()

            x = step * np.cos(gyro_all_df["angle"][gyro_i] + init_angle) + track[-1][0]
            y = step * np.sin(gyro_all_df["angle"][gyro_i] + init_angle) + track[-1][1]

            track.append(Position(x, y, results[0].z))

        # 推定結果を保存
        results.track = track

    results.plot_map()


def norm(row: pd.Series):
    return (row["x"] ** 2 + row["y"] ** 2 + row["z"] ** 2) ** 0.5


if __name__ == "__main__":
    main()
