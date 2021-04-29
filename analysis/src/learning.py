import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import time

class Network():
    """
    分類に用いるネットワーク

    Attributes
    ----------
    device : torch.device
        利用するデバイス
    model : nn.Module
        利用するモデル
    optimizer : optim.Adam
        利用する最適化手法
    criterion : nn.CrossEntropyLoss
        利用する損失関数
    """

    def __init__(self, device, num_classes):
        """
        コンストラクタ

        Parameters
        ----------
        device : torch.device
            利用するデバイス
        num_classes : int
            出力層のクラスサイズ
        """
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes) # 全結合層（FC層）の出力クラス数を変更
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def execute(self, train_loader, test_loader, max_epoch):
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
                print('=== Epoch: {} (Time: {}) ==='.format(epoch, elapsed_time))
                print('Train loss: {:.4e}, Train accuracy: {:.3f}%'.format(train_loss, train_accuracy))
                print('Test  loss: {:.4e}, Test  accuracy: {:.3f}%'.format(test_loss, test_accuracy))
                print('=================\n')

                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_params = self.model.state_dict()

        return best_params

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
            self.optimizer.zero_grad()
            # 順伝播処理
            outputs = self.model(inputs)
            # 損失関数の計算
            loss = self.criterion(outputs, targets)
            # 逆伝播
            loss.backward()
            # パラメータ更新
            self.optimizer.step()

            # 予測結果の集計
            loss_val = loss.item()
            epoch_loss += loss_val
            _, predicted = torch.max(outputs.data, 1)
            num_total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

            if batch_idx % output_interval == 0:
                    print('Train Epoch: {} [{}/{}({:.0f}%)] Loss: {:.4e}'.format(
                        epoch, num_total, len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), loss_val)
                    )

        epoch_loss /= len(train_loader)
        accuracy = 100 * correct / num_total

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
                outputs = self.model(inputs)
                # 損失関数の計算
                loss = self.criterion(outputs, targets)

                # 予測結果の集計
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                num_total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()

            epoch_loss /= len(test_loader)
            accuracy = 100 * correct / num_total

        return epoch_loss, accuracy
