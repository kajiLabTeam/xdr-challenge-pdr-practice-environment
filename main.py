from src.results import Results
from src.pdr import PDR
from src.type import Position


def main():
    pdr = PDR(maxwait=0.5)
    results = Results()

    c = 0
    for acce_df, gyro_df, acce_all_df, gyro_all_df in pdr:
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
