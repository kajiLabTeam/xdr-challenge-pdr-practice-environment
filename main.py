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
MAP_RESOLUTION = 0.01
MAP_ORIGIN_X_IN_PIXELS = 565
MAP_ORIGIN_Y_IN_PIXELS = 1480
WALKABLE_THRESHOLD = 200


def world_to_pixel(x, y):
    """実世界座標(メートル)をピクセル座標に変換する"""
    pixel_x = int(MAP_ORIGIN_X_IN_PIXELS + x / MAP_RESOLUTION)
    pixel_y = int(MAP_ORIGIN_Y_IN_PIXELS - y / MAP_RESOLUTION)
    return pixel_x, pixel_y


def pixel_to_world(pixel_x, pixel_y):
    """ピクセル座標を実世界座標(メートル)に変換する"""
    x = (pixel_x - MAP_ORIGIN_X_IN_PIXELS) * MAP_RESOLUTION
    y = (MAP_ORIGIN_Y_IN_PIXELS - pixel_y) * MAP_RESOLUTION
    return x, y


def is_walkable(px, py, map_image):
    """指定されたピクセルが通行可能かどうかを返す"""
    width, height = map_image.size
    if not (0 <= px < width and 0 <= py < height):
        return False
    return map_image.getpixel((px, py)) > WALKABLE_THRESHOLD


def map_matching(x, y, z, last_position, current_pdr_angle, map_image):
    """
    マップマッチングを実行し、位置と、補正すべき「方向」(-1:左, 1:右)を返す
    """
    px, py = world_to_pixel(x, y)
    last_px, last_py = world_to_pixel(last_position.x, last_position.y)
    
    last_walkable_px, last_walkable_py = last_px, last_py
    collision_occured = False

    dx = abs(px - last_px)
    dy = -abs(py - last_py)
    sx = 1 if last_px < px else -1
    sy = 1 if last_py < py else -1
    err = dx + dy
    temp_px, temp_py = last_px, last_py

    while True:
        if not is_walkable(temp_px, temp_py, map_image):
            collision_occured = True
            break
        last_walkable_px, last_walkable_py = temp_px, temp_py
        if temp_px == px and temp_py == py:
            break
        e2 = 2 * err
        if e2 >= dy: err += dy; temp_px += sx
        if e2 <= dx: err += dx; temp_py += sy

    new_wx, new_wy = pixel_to_world(last_walkable_px, last_walkable_py)
    corrected_position = Position(new_wx, new_wy, z)
    turn_direction = 0

    if collision_occured:
        check_px, check_py = last_walkable_px, last_walkable_py
        neighbors = {
            "E": (check_px + 1, check_py), "W": (check_px - 1, check_py),
            "N": (check_px, check_py - 1), "S": (check_px, check_py + 1)
        }
        walkable_neighbors = {d: is_walkable(x, y, map_image) for d, (x, y) in neighbors.items()}
        
        is_horz_path = walkable_neighbors["E"] or walkable_neighbors["W"]
        is_vert_path = walkable_neighbors["N"] or walkable_neighbors["S"]
        pdr_angle_deg = np.rad2deg(current_pdr_angle) % 360

        wall_angle_rad = None
        is_moving_horizontally = (45 < pdr_angle_deg < 135) or (225 < pdr_angle_deg < 315)

        if is_horz_path and is_vert_path:
            wall_angle_rad = (np.pi / 2 if 0 <= pdr_angle_deg < 180 else -np.pi / 2) if is_moving_horizontally else (0.0 if (0 <= pdr_angle_deg < 90) or (270 < pdr_angle_deg <= 360) else np.pi)
        elif is_horz_path:
            wall_angle_rad = 0.0 if (0 <= pdr_angle_deg < 90) or (270 < pdr_angle_deg <= 360) else np.pi
        elif is_vert_path:
            wall_angle_rad = np.pi / 2 if 0 <= pdr_angle_deg < 180 else -np.pi / 2

        if wall_angle_rad is not None:
            diff = wall_angle_rad - current_pdr_angle
            diff = (diff + np.pi) % (2 * np.pi) - np.pi 
            turn_direction = np.sign(diff)

    return corrected_position, turn_direction


def main():
    src_dir = Path().resolve()
    acce_file = src_dir / "data" / "acce.csv"
    gyro_file = src_dir / "data" / "gyro.csv"
    map_file = src_dir / "map" / "matching.png"
    
    try:
        map_image = Image.open(map_file).convert("L")
    except FileNotFoundError:
        print(f"エラー: マップマッチング用の画像ファイルが見つかりません: {map_file}")
        return

    # ★★★ 変更点1: DataProviderに start_timestamp=500 を指定 ★★★
    dataprovider = DataProvider(acce_file=acce_file, gyro_file=gyro_file, maxwait=0.5, offline=True, start_timestamp=625)
    
    # ★★★ 変更点2: タイムスタンプ500時点での適切な初期値を設定 ★★★
    results = Results(map_file=map_file, initial_position=Position(22.3, -5, 0.902339697))

    window_acc, window_gyro = 60, 60
    peak_distance, peak_height = 30, 1
    # ★★★ 変更点3: 初期角度も調整 ★★★
    init_angle = np.deg2rad(-140)
    
    for acce_df, gyro_df, acce_all_df, gyro_all_df in dataprovider:
        # データが空の場合はスキップ
        if acce_all_df.empty or gyro_all_df.empty:
            print("指定されたタイムスタンプ以降のデータがありません。")
            continue

        acce_fs = acce_all_df["app_timestamp"].count() / (acce_all_df["app_timestamp"].max() - acce_all_df["app_timestamp"].min())
        gyro_fs = gyro_all_df["app_timestamp"].count() / (gyro_all_df["app_timestamp"].max() - gyro_all_df["app_timestamp"].min())
        acce_all_df["norm"] = np.linalg.norm(acce_all_df[["x", "y", "z"]].values, axis=1)
        gyro_all_df["angle"] = gyro_all_df["x"].cumsum() / gyro_fs
        acce_all_df["low_norm"] = acce_all_df["norm"].rolling(window=window_acc).mean()
        peaks, _ = signal.find_peaks(acce_all_df["low_norm"], distance=peak_distance, height=peak_height)
        
        step = 0.38
        
        # ★★★ 変更点4: フィルタリングが不要になったため、シンプルなリストに戻す ★★★
        points = [results[0]]

        for p in peaks:
            time = acce_all_df["app_timestamp"][p]
            low_angle_idx = gyro_all_df["app_timestamp"].sub(time).abs().idxmin()
            
            last_position = points[-1]
            gyro_angle = gyro_all_df["angle"][low_angle_idx]
            current_pdr_angle = gyro_angle + init_angle

            x_pdr = step * np.cos(current_pdr_angle) + last_position.x
            y_pdr = step * np.sin(current_pdr_angle) + last_position.y

            matched_position, turn_direction = map_matching(x_pdr, y_pdr, results[0].z, last_position, current_pdr_angle, map_image)
            
            if turn_direction != 0:
                correction_rad = np.deg2rad(3)
                init_angle += correction_rad * turn_direction
            
            points.append(matched_position)
        
        # ★★★ 変更点5: 計算結果をそのままプロット対象とする ★★★
        results.track = points
        results.save(acce_all_df, gyro_all_df, peaks)
    
    results.plot_map()

if __name__ == "__main__":
    main()