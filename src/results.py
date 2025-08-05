"""
推定結果を簡単に保存、可視化するためのクラスです。
このファイルは触る必要はありません。
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from src.type import Position


class Results:
    map_file = "miraikan_5.bmp"
    track: list[Position] = []

    map_origin = (-5.625, -12.75)
    map_ppm = 100

    def __init__(self, map_file: str | Path, initial_position=Position(0, 0, 0)):
        """
        初期化処理
        推定結果を保存するリストを初期化する
        """
        self.track = [initial_position]
        self.bitmap_array = np.array(Image.open(map_file)) / 255.0

    def __getitem__(self, index: int) -> Position:
        """
        推定結果を取得する
        """
        return self.track[index]

    def append(self, position: Position):
        """
        推定結果を保存する
        """
        self.track.append(position)

    def reset(self):
        """
        推定結果をリセットする
        """
        self.track = []

    def to_dataframe(self):
        """
        推定結果を取得する
        """
        return pd.DataFrame(self.track)

    def plot_map(self, filename=None):
        """
        推定結果をプロットする

        Args:
            filename: 保存するファイル名。Noneの場合は表示する
        """
        df = self.to_dataframe()

        height, width = self.bitmap_array.shape[:2]

        width_m = width / self.map_ppm
        height_m = height / self.map_ppm

        extent = [
            self.map_origin[0],
            self.map_origin[0] + width_m,
            self.map_origin[1],
            self.map_origin[1] + height_m,
        ]

        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.imshow(self.bitmap_array, extent=extent, alpha=0.5, cmap="gray")
        scatter = ax.scatter(df.x, df.y, s=3, label="location (ground truth)")
        ax.set_xlabel("x (m)")

        ax.set_ylabel("y (m)")

        plt.colorbar(scatter, ax=ax, label="timestamp (s)")

        plt.legend()

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    @property
    def init_position(self) -> Position:
        """
        初期位置を返す
        オフラインモードでは、データの最初のタイムスタンプに基づく
        """
        return self[0]

    @property
    def final_position(self) -> Position:
        """
        最終位置を返す
        """
        return self.track[-1]
