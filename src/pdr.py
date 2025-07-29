"""
xDR Challenge の環境を再現するための、PDRのベースクラスを定義しています。
機能追加は pdr.py の PDR クラスで行ってください。
"""

import pandas as pd
from pathlib import Path


class PDR:
    """
    PDRのベースクラス
    データの読み込みとイテレータの実装を行う

    Attributes:
        maxwait: データを取得する最大待機時間
        max_timestamp: データの最大タイムスタンプ
        data_dir: データファイルのディレクトリ
        acce_file: 加速度センサのデータファイル名
        gyro_file: ジャイロセンサのデータファイル名
        results: 推定結果を保存するリスト
    """

    maxwait: int
    max_timestamp: int

    acce_file = "acce.csv"
    gyro_file = "gyro.csv"
    data_dir = Path().resolve() / "data"

    def __init__(self, maxwait=0.5):
        """
        初期化処理
        データを読み込む
        """
        self.maxwait = maxwait

        self.acce_df = pd.read_csv(
            self.data_dir / self.acce_file,
            sep=";",
            names=[
                "_type",
                "app_timestamp",
                "snesor_timestamp",
                "x",
                "y",
                "z",
                "accuracy",
            ],
        ).drop(columns=["_type"])
        self.gyro_df = pd.read_csv(
            self.data_dir / self.gyro_file,
            sep=";",
            names=[
                "_type",
                "app_timestamp",
                "snesor_timestamp",
                "x",
                "y",
                "z",
                "accuracy",
            ],
        ).drop(columns=["_type"])

        acc_max_timestamp = self.acce_df["snesor_timestamp"].max()
        gyro_max_timestamp = self.gyro_df["snesor_timestamp"].max()
        self.max_timestamp = min(acc_max_timestamp, gyro_max_timestamp)

    def _filter_by_timestamp(self, df, timestamp, window, only_end=True):
        """
        タイムスタンプでデータをフィルタリングする
        Args:
            df: フィルタリング対象のデータフレーム
            timestamp: フィルタリングの基準となるタイムスタンプ
            window: フィルタリングのウィンドウサイズ
            only_end: Trueの場合、ウィンドウの終端のみを返す
        Returns:
            フィルタリングされたデータフレーム
        """
        if only_end:
            return df[df["app_timestamp"] >= timestamp + window]
        else:
            # ウィンドウの開始から終了までの範囲でデータを取得
            return df[
                (df["app_timestamp"] >= timestamp)
                & (df["app_timestamp"] < timestamp + window)
            ]

    def __iter__(self):
        self.current_timestamp = 0
        return self

    def __next__(self):
        """
        イテレータの次の要素を返す
        """

        # 現在のタイムスタンプがデータの範囲を超えている場合は例外を投げる
        if self.current_timestamp >= self.max_timestamp:
            raise StopIteration

        # 新たに取得した範囲でデータを取得
        acce_df = self._filter_by_timestamp(
            self.acce_df, self.current_timestamp, self.maxwait, only_end=False
        )
        gyro_df = self._filter_by_timestamp(
            self.gyro_df, self.current_timestamp, self.maxwait, only_end=False
        )

        # 現在までの範囲でデータを取得
        acce_all_df = self._filter_by_timestamp(
            self.acce_df, self.current_timestamp, self.maxwait, only_end=True
        )
        gyro_all_df = self._filter_by_timestamp(
            self.gyro_df, self.current_timestamp, self.maxwait, only_end=True
        )

        self.current_timestamp += self.maxwait

        return acce_df, gyro_df, acce_all_df, gyro_all_df
