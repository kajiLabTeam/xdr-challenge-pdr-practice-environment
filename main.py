from pathlib import Path
import numpy as np
from scipy import signal
from scipy.linalg import norm
import matplotlib.pyplot as plt
from src.results import Results
from src.data_provider import DataProvider
from src.type import Position


def main():
    src_dir = Path().resolve()
    acce_file = src_dir / "data" / "acce.csv"
    gyro_file = src_dir / "data" / "gyro.csv"
    map_file = src_dir / "map" / "miraikan_5.bmp"

    dataprovider = DataProvider(acce_file=acce_file, gyro_file=gyro_file, maxwait=0.5,offline=True)
    results = Results(
        map_file=map_file, initial_position=Position(41.368, -10.047, 0.935)
    )

    window_acc = 60
    window_gyro = 60
    peak_distance = 30
    peak_height = 1
    init_angle = np.deg2rad(80)
    for acce_df, gyro_df, acce_all_df, gyro_all_df in dataprovider:
        acce_fs = acce_all_df["app_timestamp"].count() / (
            acce_all_df["app_timestamp"].max() - acce_all_df["app_timestamp"].min()
        )
        gyro_fs = gyro_all_df["app_timestamp"].count() / (
            gyro_all_df["app_timestamp"].max() - gyro_all_df["app_timestamp"].min()
        )

        acce_all_df["norm"] = np.linalg.norm(
            acce_all_df[["x", "y", "z"]].values, axis=1
        )

        gyro_all_df["norm"] = np.linalg.norm(
            gyro_all_df[["x", "y", "z"]].values, axis=1
        )

        gyro_all_df["angle"] = gyro_all_df["x"].cumsum() / gyro_fs

        acce_all_df["low_norm"] = acce_all_df["norm"].rolling(window=window_acc).mean()
        gyro_all_df["low_x"] = gyro_all_df["x"].rolling(window=window_gyro).mean()
        gyro_all_df["low_angle"] = (
            gyro_all_df["angle"].rolling(window=window_gyro, center=True).mean()
        )

        peaks, _ = signal.find_peaks(acce_all_df["low_norm"],distance=peak_distance, height=peak_height)
        
        # print(f"Detected peaks: {len(peaks)}")

        acce_all_df["norm"] = np.linalg.norm(
            acce_all_df[["x", "y", "z"]].values, axis=1
        )


        step = 0.4
        points = [results[0]]
        for p in peaks:
            time = acce_all_df["app_timestamp"][p]
            low_angle = gyro_all_df["app_timestamp"].sub(time).abs().idxmin()
            
            x = step * np.cos(gyro_all_df["angle"][low_angle]+init_angle ) + points[-1][0]
            y = step * np.sin(gyro_all_df["angle"][low_angle]+init_angle ) + points[-1][1]

            points.append(Position(x, y, results[0].z))
            

        results.results= points
        results.save(acce_all_df, gyro_all_df, peaks)
    
    # マップに推定結果をプロット
    results.plot_map()
    results.plot()


if __name__ == "__main__":
    main()
