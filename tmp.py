from pathlib import Path
from src.results import Results
from src.data_provider import DataProvider
from src.type import Position
import numpy as np
from scipy import signal
import pandas as pd


def main():
    src_dir = Path().resolve()
    acce_file = src_dir / "data" / "acce.csv"
    gyro_file = src_dir / "data" / "gyro.csv"
    map_file = src_dir / "map" / "miraikan_5.bmp"

    dataprovider = DataProvider(acce_file=acce_file, gyro_file=gyro_file, maxwait=0.5, offline=True)
    results = Results(map_file=map_file, initial_position=Position(41.368, -10.047, 0.935))

    #0.5秒ごとにデータが送られていることを想定して処理
    #all_をいじる
    for acce_df, gyro_df, acce_all_df, gyro_all_df in dataprovider:
        #ノルム
        acce_all_df['norm'] = acce_all_df[['x', 'y', 'z']].apply(lambda row: (row['x']**2 + row['y']**2 + row['z']**2)**(1/2), axis = 1)#todo
        gyro_all_df['norm'] = gyro_all_df[['x', 'y', 'z']].apply(lambda row: (row['x']**2 + row['y']**2 + row['z']**2)**(1/2), axis = 1)#todo
        
        #積分
        gyro_all_df['angle'] = np.cumsum(gyro_all_df['x']) / 500#todo
        
        #移動平均
        window_gyro = 200
        gyro_all_df['low_angle'] = gyro_all_df['angle'].rolling(window= window_gyro, center=True).mean()

        #ピーク検出
        peek, _ = signal.find_peaks(acce_all_df['norm'], distance=30, height=1.0)

        #ピークごとの角速度取得
        step = 0.3
        point = [Position(0,0,0)]
        for p in peek:
            x = step * np.cos(gyro_all_df['low_angle'][p]*1.2) + point[-1][0]
            y = step * np.sin(gyro_all_df['low_angle'][p]*1.2) + point[-1][1]

            point.append(Position(x, y, 0))
        results.track = point
        
        # point = pd.DataFrame(data=point, columns=['x', 'y'])

        # point['x'] = point['x'] - point['x'][0]
        # point['y'] = point['y'] - point['y'][0]
        # # 直前の推定位置
        # last_position = results[-1]

        # # 位置推定 
        # x = last_position.x + point['x']
        # y = last_position.y + point['y']
        
        # # 推定結果を保存
        # results.append(Position(x=x, y=y, z=0.5))
        

    # マップに推定結果をプロット
    results.plot_map()#todo

if __name__ == "__main__":
    main()
