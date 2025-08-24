from pathlib import Path
import numpy as np
from scipy import signal
from src.results import Results
from src.data_provider import DataProvider
from src.type import Position
import math


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
    peak_distance_sec = 0.5  # ピーク検出の最小距離（秒）
    peak_height = 1.0  # ピーク検出の最小高さ
    init_angle = np.deg2rad(68)  # 初期角度（ラジアン）
    K = 0.45  # Weinbergモデルの定数K
    angle_correction_factor = 1.2 # 角度補正係数

    for acce_df, gyro_df, acce_all_df, gyro_all_df in dataprovider:
        # サンプリング周波数の計算
        acce_fs = acce_all_df["app_timestamp"].count() / (
            acce_all_df["app_timestamp"].max() - acce_all_df["app_timestamp"].min()
        )
        gyro_fs = gyro_all_df["app_timestamp"].count() / (
            gyro_all_df["app_timestamp"].max() - gyro_all_df["app_timestamp"].min()
        )

        # ノルムの計算 (フィルタリング前の生データとして使用)
        acce_all_df["norm"] = (
            acce_all_df["x"] ** 2 + acce_all_df["y"] ** 2 + acce_all_df["z"] ** 2
        ) ** (1 / 2)

        # 角度の計算
        gyro_all_df["angle"] = np.cumsum(gyro_all_df["x"]) / gyro_fs

        # 移動平均フィルタ
        window_acc_frame = int(window_acc_sec * acce_fs)
        window_gyro_frame = int(window_gyro_sec * gyro_fs)
        acce_all_df["low_norm"] = (
            acce_all_df["norm"].rolling(window=window_acc_frame, min_periods=1).mean()
        )
        gyro_all_df["low_angle"] = (
            gyro_all_df["angle"].rolling(window=window_gyro_frame, min_periods=1).mean()
        )

        # ピークの検出（フィルタリング後のデータを使用）
        distance_frame = int(peak_distance_sec * acce_fs)
        peaks, _ = signal.find_peaks(
            acce_all_df["low_norm"],
            distance=distance_frame,
            height=peak_height,
        )

        if len(peaks) == 0:
            continue
            
        # Weinbergによる歩数推定
        detected_steps = []
        acc_norm_values = acce_all_df["norm"].values # 計算を高速化するためnumpy配列に変換
        for i in range(len(peaks)):
            # 各ピーク間の区間を設定
            start_i = peaks[i - 1] if i > 0 else 0
            end_i = peaks[i]

            # 区間内の加速度ノルムの最大値と最小値を取得
            range_acc = acc_norm_values[start_i:end_i]
            if range_acc.size == 0:
                # 区間にデータがない場合（ピークが連続した場合など）は前の歩幅を流用
                stride = detected_steps[-1] if detected_steps else 0.4
            else:
                max_acc = np.max(range_acc)
                min_acc = np.min(range_acc)
                # Weinbergの式で歩幅を計算
                stride = K * (math.pow(max_acc - min_acc, 1/4.0))

            detected_steps.append(stride)
        # 軌跡の計算
        track = [results.init_position]
        gyro_timestamps = gyro_all_df["app_timestamp"].values
        for i, peak in enumerate(peaks): # enumerateでインデックスを取得
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

            # Weinbergモデルで計算した歩幅を使用
            step = detected_steps[i]

            # 座標更新
            angle = gyro_all_df["low_angle"][gyro_i] * angle_correction_factor + init_angle
            x = step * np.cos(angle) + track[-1].x
            y = step * np.sin(angle) + track[-1].y

            track.append(Position(x, y, 0))

        # 推定結果を保存
        results.track = track

    results.plot_map()


if __name__ == "__main__":
    main()