# Candle Chart Analysis
1分足チャートを用いて、為替レートの分析を行う。

## データの取得先
[MT4 ヒストリカルデータ](https://www.axiory.com/jp/how-to-install/mt4-historical-data)

## データの取得と加工
1. 上記のサイトにおいて、2015年から2020年までの「USD / JPY」のデータを取得する。
1. 取得したファイルのうち、1年分のデータが掲載されているファイル（USDJPY_all.csv）をエディタで開き、1列目と2列目をハイフンで結合する。（結合後は、1列分データが減ることになる）

    また、年月日の部分は、ハイフンで結合する。

    ```bash
    # before
    2015.01.02,00:00,119.737,119.742,119.735,119.742,0
    2015.01.02,00:01,119.742,119.744,119.693,119.693,1
    2015.01.02,00:02,119.693,119.7,119.693,119.698,0
    2015.01.02,00:03,119.698,119.703,119.698,119.703,0
    2015.01.02,00:04,119.703,119.703,119.695,119.695,1

    # after
    2015-01-02-00:00,119.737,119.742,119.735,119.742,0
    2015-01-02-00:01,119.742,119.744,119.693,119.693,1
    2015-01-02-00:02,119.693,119.7,119.693,119.698,0
    2015-01-02-00:03,119.698,119.703,119.698,119.703,0
    2015-01-02-00:04,119.703,119.703,119.695,119.695,1
    ```

1. 編集後のファイルを1つのファイルにまとめる。

    2015年から2020年までのデータを結合すると、2,227,651行になる。

1. 以下のように、ヘッダを付加する。（ヘッダ追加後は、2,227,652行）

    ```bash
    # before
    2015-01-02-00:00,119.737,119.742,119.735,119.742,0
    2015-01-02-00:01,119.742,119.744,119.693,119.693,1
    2015-01-02-00:02,119.693,119.7,119.693,119.698,0
    2015-01-02-00:03,119.698,119.703,119.698,119.703,0
    2015-01-02-00:04,119.703,119.703,119.695,119.695,1
    (中略)
    2020-12-31-23:55,103.284,103.298,103.284,103.29,33
    2020-12-31-23:56,103.291,103.293,103.289,103.291,10
    2020-12-31-23:57,103.292,103.292,103.289,103.291,13
    2020-12-31-23:58,103.293,103.3,103.293,103.3,60
    2020-12-31-23:59,103.296,103.308,103.293,103.307,126

    # after
    datetime,open,high,low,close,volume
    2015-01-02-0:00,119.737,119.742,119.735,119.742,0
    2015-01-02-0:01,119.742,119.744,119.693,119.693,1
    2015-01-02-0:02,119.693,119.7,119.693,119.698,0
    2015-01-02-0:03,119.698,119.703,119.698,119.703,0
    2015-01-02-0:04,119.703,119.703,119.695,119.695,1
    (中略)
    2020-12-31-23:55,103.284,103.298,103.284,103.29,33
    2020-12-31-23:56,103.291,103.293,103.289,103.291,10
    2020-12-31-23:57,103.292,103.292,103.289,103.291,13
    2020-12-31-23:58,103.293,103.3,103.293,103.3,60
    2020-12-31-23:59,103.296,103.308,103.293,103.307,126
    ```

1. 作成したファイルを「concat_USDJPY2015_2020.csv」として、「analysis/csv」に格納する。

    面倒な人は、analysis/csvにそれぞれのcsvファイルを配置し、以下を実行する。csvファイル名は該当するShell Scriptを参照すること。

    ```bash
    cd analysis/csv

    chmod +x convert_csv.sh
    ./convert_csv.sh
    ```

## 起動方法とアクセス方法
ここでは、環境構築にdockerを利用する。dockerの導入方法については、割愛する。

1. docker-compose.ymlがあるディレクトリで、以下のコマンドを実行し、jupyterコンテナを起動する。

    ```bash
    # build
    docker-compose build
    # start
    docker-compose up -d
    ```

1. 起動後に、tokenの情報が必要になるため、以下を実行し、tokenを得る。

    ```bash
    chmod +x get_token.sh
    ./get_token.sh
    ```

1. jupyterコンテナを起動したPCのIPアドレスまたはドメイン名を指定し、Jupyter Notebookにアクセスする。ここで、ポート番号は「18580」を指定すること。
1. 事前に取得したtokenを入力し、Webブラウザからworkディレクトリに移動する。

## チャート画像の作成手順
### pythonファイルを直接実行する場合
1. docker containerにアクセスし、以下のコマンドを実行し、チャート画像の作成が完了するのを待つ。

    ```bash
    python create_dataset.py
    ```

### Jupyter Notebookから実行する場合
1. Jupyter Notebookから「create_candle_chart_dataset.ipynb」を起動する。
1. 起動したNotebookの先頭セルから順番に実行する。チャート画像の作成には時間がかかるため、余裕をもって実施すること。

## モデル構築の実施手順
### pythonファイルを直接実行する場合
1. docker containerにアクセスし、「src」以下に移動する。
1. 以下のコマンドを実行し、学習が完了するのを待つ。

    ```bash
    python main.py
    ```

### Jupyter Notebookから実行する場合
1. Jupyter Notebookから「deep_learning.ipynb」を起動する。
1. 起動したNotebookの先頭セルから順番に実行する。学習には時間がかかるため、余裕をもって実施すること。

    また、チャート画像が必要になるため、事前にチャート画像を作成しておくこと。
