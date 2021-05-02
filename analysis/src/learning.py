import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd.function import Function
import time

class CustomModel(nn.Module):
    """
    カスタムモデル

    Attributes
    ----------
    base_model : nn.Module
        ベースモデル
    prelu : nn.PReLU
        活性化関数
    fc : nn.Linear
        全結合層
    """

    def __init__(self, embedded_dim, num_classes):
        """
        コンストラクタ

        Parameters
        ----------
        embedded_dim : int
            埋め込みベクトルの次元数
        num_classes : int
            出力クラスサイズ
        """
        super().__init__()
        model = models.vgg16(pretrained=True)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, embedded_dim)
        self.base_model = model
        self.prelu = nn.PReLU()
        self.fc = nn.Linear(embedded_dim, num_classes)

    def forward(self, inputs):
        """
        順伝播

        Parameters
        ----------
        inputs : torch.tensor
            入力データ

        Returns
        -------
        features : torch.tensor
            埋め込みベクトル
        outputs : torch.tensor
            出力データ
        """
        features = self.prelu(self.base_model(inputs))
        outputs = self.fc(features)

        return features, outputs

class CenterLoss(nn.Module):
    """
    距離学習用の損失関数（CenterLoss）

    Attributes
    ----------
    centers : nn.Parameter
        CenterLoss用のパラメータ
    center_loss_func : CenterlossFunc
        CenterLoss計算用関数
    """

    def __init__(self, embedded_dim, num_classes):
        """
        コンストラクタ

        Parameters
        ----------
        embedded_dim : int
            埋め込みベクトルの次元数
        num_classes : int
            出力クラスサイズ
        """
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, embedded_dim))
        self.center_loss_func = CenterlossFunc.apply

    def forward(self, features, labels):
        """
        順伝播

        Parameters
        ----------
        features : torch.tensor
            埋め込みベクトル
        labels : torch.tensor
            正解ラベル

        Returns
        -------
        loss : torch.tensor
            損失関数の値
        """
        batch_size = features.size(0)
        features = features.view(batch_size, -1)
        batch_size_tensor = features.new_empty(1).fill_(batch_size)
        loss = self.center_loss_func(features, labels, self.centers, batch_size_tensor)

        return loss

class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, features, labels, centers, batch_size):
        """
        順伝播

        Parameters
        ----------
        features : torch.tensor
            埋め込みベクトル
        labels : torch.tensor
            正解ラベル
        centers : torch.tensor
            CenterLoss用のパラメータ
        batch_size : torch.tensor
            バッチサイズ

        Returns
        -------
        loss : torch.tensor
            損失関数の値
        """
        ctx.save_for_backward(features, labels, centers, batch_size)
        centers_batch = centers.index_select(0, labels.long())
        loss = (features - centers_batch).pow(2).sum() / 2.0 / batch_size

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        逆伝播

        Parameters
        ----------
        grad_output : torch.tensor
            出力層側からの勾配情報

        Returns
        -------
        outgrad_features : torch.tensor
            入力層側への勾配情報（埋め込みベクトル）
        outgrad_labels : torch.tensor
            入力層側への勾配情報（正解ラベル）
        outgrad_centers : torch.tensor
            入力層側への勾配情報（CenterLoss用のパラメータ）
        outgrad_batch_size : torch.tensor
            入力層側への勾配情報（バッチサイズ）
        """
        features, labels, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, labels.long())
        diff = centers_batch - features
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(labels.size(0))
        grad_centers = centers.new_zeros(centers.size())
        # calculate gradient
        counts = counts.scatter_add_(0, labels.long(), ones)
        grad_centers.scatter_add_(0, labels.unsqueeze(1).expand(features.size()).long(), diff)
        grad_centers = grad_centers / counts.view(-1, 1)
        # output tensor
        outgrad_features = -grad_output * diff / batch_size
        outgrad_labels = None
        outgrad_centers = grad_centers / batch_size
        outgrad_batch_size = None

        return outgrad_features, outgrad_labels, outgrad_centers, outgrad_batch_size

class Network():
    """
    分類に用いるネットワーク

    Attributes
    ----------
    device : torch.device
        利用するデバイス
    alpha : float
        損失関数のハイパーパラメータ
    model : nn.Module
        利用するモデル
    optimizer_model : optim.SGD
        利用する最適化手法（model用）
    optimizer_center_loss : optim.SGD
        利用する最適化手法（CenterLoss用）
    criterion_xentory : nn.CrossEntropyLoss
        利用する損失関数（正解ラベル用）
    criterion_center : CenterLoss
        利用する損失関数（特徴ベクトル用）
    """

    def __init__(self, device, embedded_dim, num_classes):
        """
        コンストラクタ

        Parameters
        ----------
        device : torch.device
            利用するデバイス
        embedded_dim : int
            埋め込みベクトルの次元数
        num_classes : int
            出力層のクラスサイズ
        """
        self.device = device
        self.alpha = 1e-1
        self.model = CustomModel(embedded_dim, num_classes).to(self.device)
        self.criterion_xentory = nn.CrossEntropyLoss()
        self.criterion_center = CenterLoss(embedded_dim, num_classes).to(self.device)
        self.optimizer_model = optim.SGD(self.model.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)
        self.optimizer_center_loss = optim.SGD(self.criterion_center.parameters(), lr=1e-1)

    def execute(self, train_loader, test_loader, max_epoch, result_filename):
        """
        学習とテストの実施

        Parameters
        ----------
        train_loader : DataLoader
            学習用データセット
        test_loader : DataLoader
            テスト用データセット
        max_epoch : int
            最大エポック数
        result_filename : str
            損失関数値と精度の結果の保存先

        Returns
        -------
        best_params : dict
            学習により得られた最適なパラメータ
        """
        best_accuracy = -1
        best_params = self.model.state_dict()

        with open(result_filename, 'w') as f_loss:
            f_loss.write('epoch,train_loss,train_accuracy,test_loss,test_accuracy\n')

            for epoch in np.arange(1, max_epoch + 1):
                # 学習実施
                start_time = time.time()
                train_loss, train_accuracy = self.__train(train_loader, epoch)
                elapsed_time = time.time() - start_time
                # テスト実施
                test_loss, test_accuracy = self.__test(test_loader, epoch)
                f_loss.write('{},{},{},{},{}\n'.format(epoch, train_loss, train_accuracy, test_loss, test_accuracy))
                print('=== Epoch: {} (Time: {:.3f}[sec]) ==='.format(epoch, elapsed_time))
                print('Train loss: {:.4e}, Train accuracy: {:.3%}'.format(train_loss, train_accuracy))
                print('Test  loss: {:.4e}, Test  accuracy: {:.3%}'.format(test_loss, test_accuracy))
                print('=================\n')

                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_params = self.model.state_dict()

        return best_params

    def load(self, filepath):
        """
        モデルパラメータの読み込み

        Parameters
        ----------
        filepath : str
            入力元のファイルパス
        """
        params = torch.load(filepath)
        self.model.load_state_dict(params)

    def save(self, filepath, params=None):
        """
        モデルパラメータの保存

        Parameters
        ----------
        filepath : str
            出力先のファイルパス
        params : dict
            モデルパラメータ
        """
        if params is None:
            params = self.model.state_dict()
        torch.save(params, filepath)

    def __train(self, train_loader, epoch):
        """
        学習実施

        Parameters
        ----------
        train_loader : DataLoader
            学習用データセット
        epoch : int
            エポック数

        Returns
        -------
        epoch_loss : float
            学習時の平均損失値
        accuracy : float
            正解率
        """
        # モデルを学習用に設定
        self.model.train()
        # 出力回数
        output_num = 5
        # 出力間隔
        output_interval = len(train_loader) // output_num
        if output_interval < 1:
            output_interval = 1
        # 変数の初期化
        epoch_loss = 0
        correct = 0
        num_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader, 1):
            # 利用するデバイス向けにデータを変換
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).long()

            # 勾配の初期化
            self.optimizer_model.zero_grad()
            self.optimizer_center_loss.zero_grad()
            # 順伝播処理
            features, outputs = self.model(inputs)
            # 損失関数の計算
            xentory_loss = self.criterion_xentory(outputs, targets)
            center_loss = self.criterion_center(features, targets)
            loss = xentory_loss + self.alpha * center_loss
            # 逆伝播
            loss.backward()
            # パラメータ更新
            self.optimizer_model.step()
            self.optimizer_center_loss.step()

            # 予測結果の集計
            loss_val = loss.item()
            epoch_loss += loss_val
            _, predicted = torch.max(outputs.data, 1)
            num_total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

            if batch_idx % output_interval == 0:
                    print('Train Epoch: {} [{}/{}({:.0%})] Loss: {:.4e}'.format(
                        epoch, num_total, len(train_loader.dataset), batch_idx / len(train_loader), loss_val)
                    )

        epoch_loss /= len(train_loader)
        accuracy = correct / num_total

        return epoch_loss, accuracy

    def __test(self, test_loader, epoch):
        """
        テスト実施

        Parameters
        ----------
        test_loader : DataLoader
            テスト用データセット
        epoch : int
            エポック数

        Returns
        -------
        epoch_loss : float
            テスト時の平均損失値
        accuracy : float
            正解率
        """
        with torch.no_grad():
            # モデルを評価用に設定
            self.model.eval()

            # 変数の初期化
            epoch_loss = 0
            correct = 0
            num_total = 0

            for (inputs, targets) in test_loader:
                # 利用するデバイス向けにデータを変換
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).long()

                # 順伝播処理
                features, outputs = self.model(inputs)
                # 損失関数の計算
                xentory_loss = self.criterion_xentory(outputs, targets)
                center_loss = self.criterion_center(features, targets)
                loss = xentory_loss + self.alpha * center_loss

                # 予測結果の集計
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                num_total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()

            epoch_loss /= len(test_loader)
            accuracy = correct / num_total

        return epoch_loss, accuracy
