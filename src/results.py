"""
推定結果を簡単に保存、可視化するためのクラスです。
機能追加は pdr.py の PDR クラスで行ってください。
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from src.type import Position


class Results:
    map_file = "miraikan_5.bmp"
    results: list[Position] = []

    map_dir = Path().resolve() / "map"
    bitmap_array = np.array(Image.open(map_dir / map_file)) / 255.0

    map_origin = (-5.625, -12.75)
    map_ppm = 100

    def __init__(self, initial_position=Position(0, 0, 0)):
        """
        初期化処理
        推定結果を保存するリストを初期化する
        """
        self.results = [initial_position]

    def append(self, position: Position):
        """
        推定結果を保存する
        """
        self.results.append(position)

    def reset(self):
        """
        推定結果をリセットする
        """
        self.results = []

    def to_dataframe(self):
        """
        推定結果を取得する
        """
        return pd.DataFrame(self.results)

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

    def __getitem__(self, index):
        """
        推定結果を取得する
        """
        return self.results[index]
