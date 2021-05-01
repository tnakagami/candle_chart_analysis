import os
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

# 平均値と標準偏差を算出
def estimate_mean_std(df):
    target = df.resample('H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    series = target['close'].pct_change(1).shift(-1).dropna()
    print('mean: {:.15e}'.format(series.mean()))
    print('std:  {:.15e}'.format(series.std()))

    return series

# ヒストグラムの描画
def plot_histogram(series. threshold=0.25):
    # ヒストグラムの描画
    series_std = (series - series.mean()) / series.std()
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Distribution of pct_change')
    ax.set_xlabel('pct_change')
    ax.set_ylabel('frequency')
    ax.set_ylim(0, 1.0)
    judge = series_std.abs() < threshold
    ax.hist(series_std,        bins=500, range=(-5, 5), density=True, alpha=0.3, color='b')
    ax.hist(series_std[judge], bins=500, range=(-5, 5), density=True, alpha=0.3, color='r')
    matched = series_std[judge].count()
    total = series_std.count()
    print('matched / total: {} / {} ({:.3%})'.format(matched, total, matched / total))

    return ax

# データセットの出力
def output_dataset(df, output_filename):
    df.to_csv(output_filename, header=True, index=False)

class CreateDataset():
    def __init__(self, mean, std, df, root_dir='chart'):
        self.root_dir = root_dir
        self.mean = mean
        self.std = std
        # 1分のロウソク足の読み込み
        self.df = df.copy()
        # 1時間のロウソク足の作成
        self.downsampling = df.resample('H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        self.get_middle_dir = lambda time: time.strftime('%Y/%m')

    def __get_output_path(self, time):
        output_filename = 'fig{}.png'.format(time.strftime('%Y%m%d%H%M'))
        output_path = os.path.join(self.root_dir, self.get_middle_dir(time), output_filename)

        return output_path

    # チャート出力用関数
    def plot_chart(self):
        indices = self.downsampling.index
        # 出力先のディレクトリ作成
        date = indices.to_series().apply(lambda time: self.get_middle_dir(time))
        dirs = date.drop_duplicates().to_list()
        for dirname in dirs:
            dir_path = os.path.join(self.root_dir, dirname)
            os.makedirs(dir_path, exist_ok=True)
        # figureの生成
        fig = mpf.figure(style='yahoo', figsize=(3, 3))
        # 余白の設定
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # Axesの生成
        ax = fig.add_subplot(1, 1, 1)
        # plot
        for idx in indices:
            print(idx.strftime('%Y/%m/%d %H:%M'))
            mpf.plot(self.df[idx.strftime('%Y-%m-%d %H')], type='candle', ax=ax, tight_layout=True)
            # ラベルを削除
            ax.grid(False)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.axis('off')
            fig.savefig(self.__get_output_path(idx))
            ax.clear()

        plt.close(fig)

    # チャートから正解ラベルを生成
    def create_groundtruth(self, threshold=0.25):
        # 差分抽出
        series = self.downsampling['close'].pct_change(1).shift(-1).fillna(self.mean)
        series_std = (series - self.mean) / self.std
        # DataFrameの用意
        indices = self.downsampling.index
        ret_df = pd.DataFrame(index=indices, columns=['path', 'class', 'label'])
        ret_df['path'] = indices.to_series().apply(lambda time: self.__get_output_path(time))
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
    threshold = 0.25
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
    output_train_df = train_creater.create_groundtruth(threshold)
    output_test_df = test_creater.create_groundtruth(threshold)
    output_dataset(output_train_df, train_output_filename)
    output_dataset(output_test_df, test_output_filename)

    if create_chart:
        # 時間のかかる処理
        train_creater.plot_chart()
        test_creater.plot_chart()

    # データセットのパスの確認
    for filename in [train_output_filename, test_output_filename]:
        out_df = pd.read_csv(filename)

        for row in out_df.itertuples():
            if not os.path.exists(row.path):
                print('{} does not exist'.format(row.path))
