import numpy as np
import random
import torch
import os
from dataloader import CreateDataLoader
from learning import Network

if __name__ == '__main__':
    seed = 1
    # デバイス名と乱数の初期値の設定
    np.random.seed(seed=seed)
    random.seed(seed)

    if torch.cuda.is_available():
        device_name = 'cuda'
        torch.cuda.manual_seed_all(seed)
    else:
        device_name = 'cpu'
        torch.cuda.manual_seed(seed)
    device = torch.device(device_name)

    # ==================
    # データセットの定義
    # ==================
    root_dir = '..'
    train_file_path = os.path.join(root_dir, 'train_dataset.csv')
    test_file_path = os.path.join(root_dir, 'test_dataset.csv')
    dataloader_creater = CreateDataLoader(root_dir=root_dir, batch_size=128)
    train_loader = dataloader_creater.get_dataloader(train_file_path, is_shuffle=True)
    test_loader = dataloader_creater.get_dataloader(test_file_path, is_shuffle=False)

    # =============
    # Networkの定義
    # =============
    num_classes = 3
    net = Network(device, num_classes)

    # ================
    # 学習と評価の実施
    # ================
    max_epoch = 2
    result_filename = 'result_loss.csv'
    best_params = net.execute(train_loader, test_loader, max_epoch, result_filename)

    # モデルの出力
    print(net.model)
    # ======================
    # モデルパラメータの保存
    # ======================
    torch.save(best_params, 'resnet18.pth')
