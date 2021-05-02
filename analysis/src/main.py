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
    train_filename = os.path.join(root_dir, 'train_dataset.csv')
    test_filename = os.path.join(root_dir, 'test_dataset.csv')
    dataloader_creater = CreateDataLoader(root_dir=root_dir, batch_size=128)
    train_loader = dataloader_creater.get_dataloader(train_filename, is_shuffle=True)
    test_loader = dataloader_creater.get_dataloader(test_filename, is_shuffle=False)

    # =============
    # Networkの定義
    # =============
    embedded_dim = 128
    num_classes = 3
    net = Network(device, embedded_dim, num_classes)
    result_filename = os.path.join(root_dir, 'result_loss.csv')
    model_path = os.path.join(root_dir, '{}.pth'.format(net.model.__class__.__name__.lower()))
    # パラメータの読み込み
    if os.path.exists(model_path):
        net.load(model_path)

    # ================
    # 学習と評価の実施
    # ================
    max_epoch = 2
    best_params = net.execute(train_loader, test_loader, max_epoch, result_filename)

    # モデルの出力
    print(net.model)
    # ======================
    # モデルパラメータの保存
    # ======================
    net.save(model_path, params=best_params)