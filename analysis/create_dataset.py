import os
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib import gridspec

# 平均値と標準偏差を算出
def estimate_mean_std(df):
    target = df.resample('H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    series = target['close'].pct_change(1).shift(-1).fillna(0)
    print('mean: {:.15e}'.format(series.mean()))
    print('std:  {:.15e}'.format(series.std()))

    return series

# ヒストグラムの描画
def plot_histogram(series. threshold=0.1):
    # ヒストグラムの描画
    series_std = (series - series.mean()) / series.std()
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Distribution of pct_change')
    ax.set_xlabel('pct_change')
    ax.set_ylabel('frequency')
    ax.set_ylim(0, 1.0)
    judge = series_std.abs() < threshold
    ax.hist(series_std,        bins=1000, range=(-5, 5), density=True, alpha=0.3, color='b')
    ax.hist(series_std[judge], bins=1000, range=(-5, 5), density=True, alpha=0.3, color='r')
    ax.grid()
    matched = series_std[judge].count()
    total = series_std.count()
    print('matched / total: {} / {} ({:.3%})'.format(matched, total, matched / total))

    return ax

# データセットの出力
def output_dataset(df, output_filename):
    df.to_csv(output_filename, header=True, index=False)

class CreateDataset():
    def __init__(self, mean, std, df):
        self.mean = mean
        self.std = std
        # 1分のロウソク足の読み込み
        self.df = df.copy()
        # 1時間のロウソク足の作成
        self.downsampling = df.resample('H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        self.sampling_index = pd.Series([idx for idx in self.downsampling.index if idx in df.index])

    # チャート出力用関数
    def plot_chart(self):
        # figureの生成
        fig = mpf.figure(style='yahoo', figsize=(3, 3))
        # 余白の設定
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # GridSpecの設定
        spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[2, 1])
        spec.update(wspace=0.025, hspace=0) # 余白の更新
        # Axesの生成
        chart_ax = fig.add_subplot(spec[0])
        volume_ax = fig.add_subplot(spec[1])
        # plot
        indices = pd.to_datetime(self.sampling_index.tolist())
        kwargs = {'type': 'candle', 'volume': volume_ax, 'ax': chart_ax, 'tight_layout': True}
        for idx in indices:
            print(idx.strftime('%Y/%m/%d %H:%M'))
            mpf.plot(self.df[idx.strftime('%Y-%m-%d %H')], **kwargs)
            # ラベルを削除
            for ax in [chart_ax, volume_ax]:
                ax.grid(False)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                ax.axis('off')
            fig.savefig('chart/fig{}.png'.format(idx.strftime('%Y%m%d%H%M')))
            chart_ax.clear()
            volume_ax.clear()

        plt.close(fig)

    # チャートから正解ラベルを生成
    def create_groundtruth(self, threshold=0.1):
        # 差分抽出
        series = self.downsampling['close'].pct_change(1).shift(-1).fillna(0)
        series_std = (series - self.mean) / self.std
        # DataFrameの用意
        indices = pd.to_datetime(self.sampling_index.tolist())
        ret_df = pd.DataFrame(index=indices, columns=['path', 'class', 'label'])
        ret_df['path'] = indices.to_series().apply(lambda time: 'chart/fig{}.png'.format(time.strftime('%Y%m%d%H%M')))
        # 初期化（下降）
        ret_df['class'] = 0
        ret_df['label'] = 'down'
        # 「停滞する」部分の抽出
        judge = series_std.abs() < threshold
        ret_df.loc[judge, 'class'] = 1
        ret_df.loc[judge, 'label'] = 'stay'
        # 「上昇する」部分の抽出
        judge = series_std >= threshold
        ret_df.loc[judge, 'class'] = 2
        ret_df.loc[judge, 'label'] = 'up'

        return ret_df

if __name__ == '__main__':
    # データの読み込み
    df = pd.read_csv('csv/concat_USDJPY2015_2020.csv', index_col='datetime', parse_dates=True)
    create_chart = True

    # ================
    # データセット作成
    # ================
    threshold = 0.1
    train_output_filename = 'train_dataset.csv'
    test_output_filename = 'test_dataset.csv'
    train_df = df[:'2019'].copy()
    test_df = df['2020'].copy()

    # データの推定
    series = estimate_mean_std(df)
    plot_histogram(series, threshold)

    # インスタンス生成
    mean, std = series.mean(), series.std()
    train_creater = CreateDataset(mean, std, train_df)
    test_creater = CreateDataset(mean, std, test_df)
    # 正解ラベルの生成
    output_train_df = train_creater.create_groundtruth()
    output_test_df = test_creater.create_groundtruth()
    output_dataset(output_train_df, train_output_filename)
    output_dataset(output_test_df, test_output_filename)

    if create_chart:
        # 時間のかかる処理
        train_creater.plot_chart(threshold)
        test_creater.plot_chart(threshold)

    # データセットのパスの確認
    for filename in [train_output_filename, test_output_filename]:
        out_df = pd.read_csv(filename)

        for row in out_df.itertuples():
            if not os.path.exists(row.path):
                print('{} does not exist'.format(row.path))
