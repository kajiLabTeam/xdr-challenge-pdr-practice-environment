# xDR Challenge 環境を再現した PDR 環境

xDR Challenge で PDR を実装するための環境を再現したものです。
xDR Challenge では 0.5s ごとに送られてくるデータをリアルタイムで位置推定する必要があります。
この環境では、加速度センサとジャイロセンサのデータを使用できる範囲(未来を除く過去~現在の範囲)だけを返すようにしています。

## セットアップ

仮想環境の作成

```bash
python -m venv venv
source venv/bin/activate
```

依存関係のインストール

```bash
pip install -r requirements.txt
```

## 使い方

```py
dataprovider = DataProvider(maxwait=0.5)
results = Results()
```

xDR Challenge では 0.5s ごとにデータを取得し、位置推定を行います。
`maxwait` パラメータを指定することで、取得データの間隔を調整できます。

- `dataprovider`: データを取得するためのイテレータ
- `results`: 位置推定の結果を格納するためのオブジェクト

```py
for acce_df, gyro_df, acce_all_df, gyro_all_df in dataprovider:
```

- `acce_df`: 新たに取得した加速度
- `gyro_df`: 新たに取得した角速度
- `acce_all_df`: 過去から現在までの加速度
- `gyro_all_df`: 過去から現在までの角速度

```py
position = Position(x=x, y=y, z=z)
```

位置を表す名前付きタプルを作成します。

```py
results.append(position)
```

位置推定の結果を保存します。

```py
results.plot_map()
# results.plot_map("output_map.png")
```

位置推定の結果をマップ上にプロットします。
ファイル名を指定することで、画像ファイルとして保存することもできます。
