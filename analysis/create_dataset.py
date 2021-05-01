import os
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
class AnalysisCandleChart:
    """
    ローソク足チャートの分析

    Attributes
    ----------
    grouping_rule : str
        チャートをグループ化する際のルール
        デフォルト：4時間ごと
    pca : PCA
        主成分分析処理用のインスタンス
    """

    def __init__(self, grouping_rule='4H', pca=None):
        """
        コンストラクタ

        Parameters
        ----------
        grouping_rule : str
            チャートをグループ化する際のルール
            デフォルト：4時間ごと
        pca : PCA
            主成分分析処理用のインスタンス
        """
        self.grouping_rule = grouping_rule
        self.pca = pca

    def downsampling(self, df, rule):
        """
        ダウンサンプリング処理

        Parameters
        ----------
        df : DataFrame
            ダウンサンプリング対象のDataFrame
        rule : str
            サンプリング時のルール

        Returns
        -------
        downsampling_df : DataFrame
            ダウンサンプリングされたDataFrame
        """
        downsampling_df = df.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

        return downsampling_df

    def diff_func(self, series):
        """
        変化量の算出処理

        Parameters
        ----------
        series : Series
            処理対象のSeries

        Returns
        -------
        diff : Series
            変化量
        """
        diff = series.pct_change(1).shift(-1)
        diff = diff.fillna(diff.mean())

        return diff

    def estimate_diff(self, df):
        """
        変化量の推定処理

        Parameters
        ----------
        df : DataFrame
            処理対象のDataFrame
        pca : PCA
            次元圧縮に利用する

        Returns
        -------
        diff : Series
            推定した変化量
        """
        # ダウンサンプリングしたローソク足チャートの作成
        dfs = {
            'hour':      self.downsampling(df, 'H'),  # 1時間のローソク足チャート
            'one_sixth': self.downsampling(df, '4H'), # 4時間のローソク足チャート
            '1day':      self.downsampling(df, 'D'),  # 1日のローソク足チャート
            '3days':     self.downsampling(df, '3D'), # 3日のローソク足チャート
        }
        # データごとに変化量を計算
        diffs = {key: self.diff_func(target['close']) for key, target in dfs.items()}
        # インデックスの作成
        indices = pd.concat([series.index.to_series() for series in diffs.values()]).drop_duplicates().sort_values()
        # DataFrameの作成
        diff_df = pd.DataFrame(data=np.nan, index=indices, columns=list(dfs.keys()))
        # データが存在する部分を更新
        for key, series in diffs.items():
            index = series.index.to_series().apply(lambda time: time.strftime('%Y/%m/%d %H:%M'))
            diff_df.loc[index, key] = series
        # 線形補間
        diff_df = diff_df.interpolate()
        # ダウンサンプリングした結果から変化量を推定
        if self.pca is None:
            self.pca = PCA(n_components=1)
            self.pca.fit(diff_df)
        feature = self.pca.transform(diff_df)
        series = pd.Series(feature.flatten(), index=diff_df.index)
        # グループ化する際のルールを用いて、平均的な変化量を算出
        diff = series.resample(self.grouping_rule).mean()

        return diff

class CreateDataset(AnalysisCandleChart):
    """
    データセットの作成

    Attributes
    ----------
    root_dir : str
        データセットのルートディレクトリ
    chart_rule : str
        出力するチャートのルール
        デフォルト：10分毎
    threshold : float
        変化量の閾値
        デフォルト：0.1
    mean : float
        変化量の平均値
    std : float
        変化量の標準偏差
    """

    def __init__(self, diff, root_dir='chart', chart_rule='10T', threshold=0.1, **kwargs):
        """
        コンストラクタ

        Parameters
        ----------
        diff : Series
            推定した変化量
        root_dir : str
            データセットのルートディレクトリ
        chart_rule : str
            出力するチャートのルール
            デフォルト：10分毎
        threshold : float
            変化量の閾値
            デフォルト：0.1
        """
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.chart_rule = chart_rule
        self.threshold = threshold
        self.mean = diff.mean()
        self.std = diff.std()

    def __get_basedir(self, timestamp):
        """
        チャート格納先のbase directory

        Parameters
        ----------
        timestamp : datetime
            チャートの時間情報

        Returns
        -------
        base_dir : str
            base directory
        """
        base_dir = os.path.join(self.root_dir, timestamp.strftime('%Y/%m'))

        return base_dir

    def __get_output_path(self, timestamp):
        """
        出力ファイルパスの取得

        Parameters
        ----------
        timestamp : datetime
            チャートの時間情報

        Returns
        -------
        output_path : str
            出力ファイルパス
        """
        output_filename = 'fig{}.png'.format(timestamp.strftime('%Y%m%d%H%M'))
        output_path = os.path.join(self.__get_basedir(timestamp), output_filename)

        return output_path

    def output_candle_chart(self, df):
        """
        ローソク足チャートの出力

        Parameters
        ----------
        df : DataFrame
            処理対象のDataFrame
        """
        # チャートを表示する間隔でダウンサンプリング
        downsampling_df = self.downsampling(df, self.chart_rule)
        # 出力先のディレクトリ作成
        dir_pathes = downsampling_df.index.to_series().apply(self.__get_basedir).drop_duplicates().to_list()
        for dir_path in dir_pathes:
            os.makedirs(dir_path, exist_ok=True)
        # figureの生成
        fig = mpf.figure(style='yahoo', figsize=(3, 3))
        # 余白の設定
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # Axesの生成
        ax = fig.add_subplot(1, 1, 1)
        # グルーピングしてプロット
        grouped = filter(lambda xs: not xs[1].empty, downsampling_df.groupby(pd.Grouper(freq=self.grouping_rule)))
        for timestamp, target_df in grouped:
            print(timestamp.strftime('%Y/%m/%d %H:%M'))
            mpf.plot(target_df, type='candle', ax=ax, tight_layout=True)
            # ラベルを削除
            ax.grid(False)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.axis('off')
            fig.savefig(self.__get_output_path(timestamp))
            ax.clear()

        plt.close(fig)

    # チャートから正解ラベルを生成
    def create_groundtruth(self, df):
        # 変化量の取得
        series = self.estimate_diff(df)
        series_std = (series - self.mean) / self.std
        # DataFrameの用意
        indices = self.downsampling(df, self.grouping_rule).index
        ret_df = pd.DataFrame(index=indices, columns=['path', 'class', 'label'])
        ret_df['path'] = indices.to_series().apply(self.__get_output_path)
        # 初期化（下降）
        ret_df['class'] = 0
        ret_df['label'] = 'down'
        # 「停滞する」部分の抽出
        judge = series_std.abs() < self.threshold
        ret_df.loc[judge, 'class'] = 1
        ret_df.loc[judge, 'label'] = 'stay'
        # 「上昇する」部分の抽出
        judge = series_std >= self.threshold
        ret_df.loc[judge, 'class'] = 2
        ret_df.loc[judge, 'label'] = 'up'

        return ret_df

    # データセットの出力
    def output_dataset(self, df, output_filename):
        df.to_csv(output_filename, header=True, index=False)

# ヒストグラムの描画
def plot_histogram(diff, mean, std, threshold=0.1):
    """
    ヒストグラムの描画

    Parameters
    ----------
    diff : Series
        推定した変化量
    mean : float
        平均値
    std : float
        標準偏差
    threshold : float
        変化量の閾値
        デフォルト：0.1
    """
    diff_std = (diff - mean) / std
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Distribution of pct_change')
    ax.set_xlabel('pct_change')
    ax.set_ylabel('frequency')
    ax.set_ylim(0, 1.0)
    ax.grid(False)
    judge = diff_std.abs() < threshold
    ax.hist(diff_std,        bins=500, range=(-5,5), density=True, alpha=0.3, color='b')
    ax.hist(diff_std[judge], bins=500, range=(-5,5), density=True, alpha=0.3, color='r')
    # 閾値以内に入っているデータをカウント
    matched = diff_std[judge].count()
    total = diff_std.count()
    print('matched / total: {} / {} ({:.3%})'.format(matched, total, matched / total))

    return ax

if __name__ == '__main__':
    # データの読み込み
    df = pd.read_csv('csv/concat_USDJPY2015_2020.csv', index_col='datetime', parse_dates=True)

    # ================
    # データセット作成
    # ================
    create_chart = True
    threshold = 0.3
    train_output_filename = 'train_dataset.csv'
    test_output_filename = 'test_dataset.csv'
    train_df = df[:'2019'].copy()
    test_df = df['2020'].copy()

    # 推定処理
    analyzer = AnalysisCandleChart()
    diff = analyzer.estimate_diff(df)
    mean, std = diff.mean(), diff.std()
    print('mean: {:.15e}'.format(mean))
    print('std:  {:.15e}'.format(std))
    plot_histogram(diff, mean, std, threshold=threshold)

    # データセットの生成
    creater = CreateDataset(diff, threshold=threshold, pca=analyzer.pca)
    creater.output_dataset(creater.create_groundtruth(train_df), train_output_filename)
    creater.output_dataset(creater.create_groundtruth(test_df), test_output_filename)

    if create_chart:
        creater.output_candle_chart(train_df)
        creater.output_candle_chart(test_df)

    # データセットのパスの確認
    for filename in [train_output_filename, test_output_filename]:
        out_df = pd.read_csv(filename)

        for row in out_df.itertuples():
            if not os.path.exists(row.path):
                print('{} does not exist'.format(row.path))
