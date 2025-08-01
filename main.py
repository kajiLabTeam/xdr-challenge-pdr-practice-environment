from pathlib import Path

import numpy as np
from src.results import Results
from src.data_provider import DataProvider
from src.type import Position


def main():
    src_dir = Path().resolve()
    acce_file = src_dir / "data" / "acce.csv"
    gyro_file = src_dir / "data" / "gyro.csv"
    map_file = src_dir / "map" / "miraikan_5.bmp"

    dataprovider = DataProvider(acce_file=acce_file, gyro_file=gyro_file, maxwait=0.5)
    results = Results(map_file=map_file, initial_position=Position(0, 0, 0))

    fs = 50  # サンプリング周波数 (Hz) の例。必要に応じて正しい値に変更してください。
    for acce_df, gyro_df, acce_all_df, gyro_all_df in dataprovider:
        acce_all_df["norm"] = acce_all_df[["x", "y", "z"]].apply(
            lambda row: (row["x"] ** 2 + row["y"] ** 2 + row["z"] ** 2) ** 0.5, axis=1
        )
        
        acce_all_df['angle']= np.cumsum(gyro_all_df['x']) / fs

            # 直前の推定位置
        last_position = results[-1]

        # 位置推定
        x = last_position.x + acce_df["x"].mean()
        y = last_position.y + acce_df["y"].mean()
        z = last_position.z

        # 推定結果を保存
        results.append(Position(x=x, y=y, z=z))
        

    # マップに推定結果をプロット
    results.plot_map()


if __name__ == "__main__":
    main()
