import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os

class _CsvFileDataset(Dataset):
    """
    csvデータからデータセットの読み込み

    Attributes
    ----------
    transform : torchvision.transforms
        前処理用の関数
    pathes : list of str
        画像データのpathのリスト
    targets : list of int
        クラスラベルのリスト
    """

    def __init__(self, csv_file, root_dir='.', transform=None):
        """
        コンストラクタ

        Parameters
        ----------
        csv_file : str (extension: .csv)
            csvファイル
        transform : torchvision.transforms
            前処理用の関数
        """
        self.transform = transform
        self.pathes, self.targets = self.__make_dataset(root_dir, csv_file)

    def __make_dataset(self, root_dir, csv_file):
        """
        データセットの作成

        Parameters
        ----------
        root_dir : str
            ルートディレクトリ
        csv_file : str
            csvファイル

        Returns
        -------
        samples : list of tuple
            (画像データのpath, label)のリスト
        """
        # csvファイルの読み込み
        df = pd.read_csv(csv_file, engine='python')
        df['target_path'] = df['path'].map(lambda path: os.path.join(root_dir, path))
        # 画像データのpathとclassを取得（有効なpathのみ）
        target = df[df['target_path'].apply(lambda path: os.path.exists(path))].copy()
        pathes = target['target_path'].to_list()
        targets = list(filter(int, target['class'].to_list()))

        return pathes, targets

    def __getitem__(self, index):
        """
        画像データの取得

        Parameters
        ----------
        index : int
            データ取得時のインデックス

        Returns
        -------
        sample : torch.Tensor
            画像データ
        target : int
            ラベルデータ
        """
        path, target = self.pathes[index], self.targets[index]
        # PIL形式で画像データを読み込む
        img = Image.open(path)
        sample = img.convert('RGB')
        # 変換
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        """
        データセットのサイズの取得

        Returns
        -------
        data_size : int
            データセットのサイズ
        """
        data_size = len(self.targets)

        return data_size

class CreateDataLoader():
    """
    データローダの生成

    Attributes
    ----------
    root_dir : str
        ルートディレクトリ
    batch_size : int
        バッチサイズ
    transform : torchvision.transform
        前処理用の関数
    """

    def __init__(self, root_dir='.', batch_size=64):
        """
        コンストラクタ
        Parameters
        ----------
        batch_size
            バッチサイズ
        """
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

    def get_dataloader(self, csv_file, is_shuffle=False, num_workers=2):
        """
        データローダの取得

        Parameters
        ----------
        csv_metafile : str
            データセットのcsvファイル名
        is_shuffle : bool
            シャッフルの有無
        num_workers : int
            並列数

        Returns
        -------
        dataloader : DataLoader
            データセットのデータローダ
        """
        # データセットの定義
        dataset = _CsvFileDataset(csv_file, root_dir=self.root_dir, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=is_shuffle, num_workers=num_workers, pin_memory=True)

        return dataloader
