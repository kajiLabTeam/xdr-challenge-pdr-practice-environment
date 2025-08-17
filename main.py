from pathlib import Path
import numpy as np
from scipy import signal
from scipy.linalg import norm
import matplotlib.pyplot as plt
from src.results import Results
from src.data_provider import DataProvider
from src.type import Position
from PIL import Image


# --- マップマッチング設定 ---

# 1. マップ画像の解像度 (1ピクセルあたりのメートル)
# 例: 0.1m/pixel -> 1ピクセルが10cm四方に相当
MAP_RESOLUTION = 0.1 # この値を実際のマップに合わせて調整してください

# 2. マップの原点ピクセル座標 (PDR座標系の(0,0)がマップ画像のどのピクセルに対応するか)
# 左上を(0,0)とします。
MAP_ORIGIN_X_IN_PIXELS = 4525
MAP_ORIGIN_Y_IN_PIXELS = 2653

# 3. 通行可能エリアの色の閾値 (0:黒 〜 255:白)
# グレースケール画像と仮定。この値より大きい輝度値を持つピクセルを通行可能とみなす。
WALKABLE_THRESHOLD = 200

# 4. 最近傍探索範囲 (ピクセル)
# 推定位置が壁の中だった場合に、この範囲で最も近い通行可能エリアを探す
NEAREST_SEARCH_RANGE = 20 # ピクセル単位

def world_to_pixel(x, y):
    """実世界座標(メートル)をピクセル座標に変換する"""
    pixel_x = int(MAP_ORIGIN_X_IN_PIXELS + x / MAP_RESOLUTION)
    pixel_y = int(MAP_ORIGIN_Y_IN_PIXELS - y / MAP_RESOLUTION) # Y軸は逆
    return pixel_x, pixel_y

def pixel_to_world(pixel_x, pixel_y):
    """ピクセル座標を実世界座標(メートル)に変換する"""
    x = (pixel_x - MAP_ORIGIN_X_IN_PIXELS) * MAP_RESOLUTION
    y = (MAP_ORIGIN_Y_IN_PIXELS - pixel_y) * MAP_RESOLUTION # Y軸は逆
    return x, y

def map_matching(x, y, z, map_image):
    """マップマッチングを実行して位置を補正する"""
    px, py = world_to_pixel(x, y)
    width, height = map_image.size

    # マップ範囲外なら何もしない
    if not (0 <= px < width and 0 <= py < height):
        return Position(x, y, z)

    # 現在位置のピクセルの輝度値を取得
    pixel_value = map_image.getpixel((px, py))

    # 通行可能エリアなら位置をそのまま返す
    if pixel_value > WALKABLE_THRESHOLD:
        return Position(x, y, z)

    # --- 通行不可エリアの場合、最も近い通行可能エリアを探す ---
    min_dist = float('inf')
    best_pos = Position(x, y, z)
    
    # 指定された範囲で探索
    for i in range(-NEAREST_SEARCH_RANGE, NEAREST_SEARCH_RANGE + 1):
        for j in range(-NEAREST_SEARCH_RANGE, NEAREST_SEARCH_RANGE + 1):
            search_px, search_py = px + i, py + j
            
            # マップ範囲内かチェック
            if not (0 <= search_px < width and 0 <= search_py < height):
                continue
            
            # 通行可能かチェック
            if map_image.getpixel((search_px, search_py)) > WALKABLE_THRESHOLD:
                dist = i**2 + j**2 # 距離の2乗で比較
                if dist < min_dist:
                    min_dist = dist
                    wx, wy = pixel_to_world(search_px, search_py)
                    best_pos = Position(wx, wy, z)

    return best_pos



def main():
    src_dir = Path().resolve()
    acce_file = src_dir / "data" / "acce.csv"
    gyro_file = src_dir / "data" / "gyro.csv"
    map_file = src_dir / "map" / "matching.png"
    
    # マップマッチング用の画像ファイルをロード
    matching_map_file = src_dir / "map" / "matching.png"
    try:
        # グレースケールで読み込む
        map_image = Image.open(matching_map_file).convert("L")
    except FileNotFoundError:
        print(f"エラー: マップマッチング用の画像ファイルが見つかりません: {matching_map_file}")
        return


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

        acce_all_df["norm"] = np.linalg.norm(
            acce_all_df[["x", "y", "z"]].values, axis=1
        )
        

        step = 0.38
        points = [results[0]]
        for p in peaks:
            time = acce_all_df["app_timestamp"][p]
            low_angle = gyro_all_df["app_timestamp"].sub(time).abs().idxmin()
            
            # PDRによる位置推定
            x_pdr = step * np.cos(gyro_all_df["angle"][low_angle]+init_angle ) + points[-1][0]
            y_pdr = step * np.sin(gyro_all_df["angle"][low_angle]+init_angle ) + points[-1][1]

            # マップマッチングによる位置補正
            matched_position = map_matching(x_pdr, y_pdr, results[0].z, map_image)
            points.append(matched_position)
            # points.append(Position(x_pdr, y_pdr, results[0].z))
            

        results.results= points
        results.save(acce_all_df, gyro_all_df, peaks)
    
    # マップに推定結果をプロット
    results.plot_map()



if __name__ == "__main__":
    main()
