import os
import numpy as np
import random
import torch
import urllib.request
from pathlib import Path
import zipfile

from data_loader import DataLoader
from data_processor import DataProcessor
from patch_extractor import PatchExtractor
from data_splitter import DataSplitter
from trainer import Trainer
from feature_extractor import FullyConnectedFeatureExtractor, MLPViTHybridFeatureExtractor, VAE_Wrapper

def main():
    def _srs(s):
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s); torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    def _download_file(url, filepath, verbose=False):
        if os.path.exists(filepath):
            if verbose:
                print(f"本地加载: {os.path.basename(filepath)}")
            return
        if verbose:
            print(f"下载文件: {os.path.basename(filepath)}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            if verbose:
                def _show_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    if total_size > 0:
                        percent = min(100, downloaded * 100 / total_size)
                        bar_length = 40
                        filled = int(bar_length * downloaded / total_size)
                        bar = '=' * filled + '-' * (bar_length - filled)
                        print(f'\r[{bar}] {percent:.1f}% ({downloaded}/{total_size} bytes)', end='', flush=True)
                    else:
                        print(f'\r已下载: {downloaded} bytes', end='', flush=True)
                urllib.request.urlretrieve(url, filepath, _show_progress)
                print(f'\n下载完成: {os.path.basename(filepath)}')
            else:
                urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            if verbose:
                print()
            raise RuntimeError(f"下载文件失败: {url} -> {filepath}, 错误: {e}")
    def _download_dataset(dataset_name, dataset_path, base_url):
        dataset_dir = os.path.join(dataset_path, dataset_name)
        if dataset_name == 'HS':
            required_files = ['Houston.mat', 'Houston_LR.mat', 'Houston_gt.mat']
        else:
            return
        if os.path.exists(dataset_dir) and all(os.path.exists(os.path.join(dataset_dir, f)) for f in required_files):
            print(f"本地加载数据集: {dataset_name}")
            return
        print(f"下载数据集: {dataset_name}")
        os.makedirs(dataset_dir, exist_ok=True)
        for filename in required_files:
            filepath = os.path.join(dataset_dir, filename)
            if not os.path.exists(filepath):
                url = base_url + filename
                _download_file(url, filepath, verbose=True)
    dataset_root = os.path.join('C:', os.sep, 'code', 'dataset')
    model_dir = os.path.join('C:', os.sep, 'code', 'saved_models')
    data_base_url = 'https://github.com/zwdmw/RL-FM1/releases/download/model/'
    model_base_url = 'https://github.com/zwdmw/RL-FM1/releases/download/model/'
    dataset_name = 'HS'
    norm_type = 'scale'
    extractor_type = 'fully_connected'
    compute_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_value = 4547029
    embed_size = 256
    hidden_size = 256
    patch_dim = 7
    train_samples = 10
    layer_count = 1
    batch_count = 300
    neighbor_count = 1
    latent_size = 256
    timestep_count = 10
    episode_count = 300
    update_epochs = 3
    learn_rate = 1e-4
    _srs(seed_value)
    _download_dataset(dataset_name, dataset_root, data_base_url)
    l1 = DataLoader(dataset_root, dataset_name); a1, a2, a3 = l1.load_data()
    p1 = DataProcessor(scaler_method=norm_type); a4, a5 = p1.process_data(a1, a2)
    b1 = int(np.unique(a3)[0])
    e1 = PatchExtractor(patch_size=patch_dim, background_label=b1); a6, a7, a8 = e1.extract_patches(a4, a5, a3)
    s1 = DataSplitter(random_seed=seed_value, background_label=b1); t1, t2 = s1.split_data(a6, a7, a8, samples_per_class=train_samples)
    tr = Trainer(device=compute_device, random_seed=seed_value)
    _x1 = lambda a, b, c, d, e, f, g: ((FullyConnectedFeatureExtractor if a == extractor_type else MLPViTHybridFeatureExtractor)(in_channels=b, feature_dim=c, dropout=0.1, random_seed=d) if a == extractor_type else (FullyConnectedFeatureExtractor if a == extractor_type else MLPViTHybridFeatureExtractor)(in_channels=b, patch_size=e, feature_dim=c, depth=f, num_heads=4, mlp_ratio=4., dropout=0.1, random_seed=d)).to(g)
    _x2 = lambda t, m, p: t.load_model(m, p)
    i1, i2 = t1['hsi'].shape[3], t1['lidar'].shape[3]
    hsi_model_path = os.path.join(model_dir, 'hsi_model.pth')
    lidar_model_path = os.path.join(model_dir, 'lidar_model.pth')
    _download_file(model_base_url + 'hsi_model.pth', hsi_model_path)
    _download_file(model_base_url + 'lidar_model.pth', lidar_model_path)
    _f1 = lambda: _x2(tr, _x1(extractor_type, i1, embed_size, seed_value, patch_dim, layer_count, compute_device), hsi_model_path)
    _f2 = lambda: _x2(tr, _x1(extractor_type, i2, embed_size, seed_value, patch_dim, layer_count, compute_device), lidar_model_path)
    m1, m2 = _f1(), _f2()
    f1, f2 = tr.extract_features(m1, t1['hsi']), tr.extract_features(m2, t1['lidar'])
    f3, f4 = tr.extract_features(m1, t2['hsi']), tr.extract_features(m2, t2['lidar'])
    hsi_vae_path = os.path.join(model_dir, 'hsi_vae_model_latest.pth')
    lidar_vae_path = os.path.join(model_dir, 'lidar_vae_model_latest.pth')
    _download_file(model_base_url + 'hsi_vae_model_latest.pth', hsi_vae_path)
    _download_file(model_base_url + 'lidar_vae_model_latest.pth', lidar_vae_path)
    _x3 = lambda d1, d2: (_x2(tr, VAE_Wrapper(input_dim=d1, latent_dim=latent_size, hidden_dims=[128, 64], random_seed=seed_value).to(compute_device), hsi_vae_path), _x2(tr, VAE_Wrapper(input_dim=d2, latent_dim=latent_size, hidden_dims=[128, 64], random_seed=seed_value).to(compute_device), lidar_vae_path))
    v1, v2 = _x3(f1.shape[1], f2.shape[1])
    ag, _, _ = tr.train_rl_policy(m1, m2, v1, v2, t1, t2, num_timesteps=timestep_count, num_episodes=episode_count, ppo_epochs=update_epochs, batch_size=batch_count, learning_rate=learn_rate, knn_neighbors=neighbor_count, save_models=True, model_save_path=model_dir)
    f5, f6 = tr.extract_features(m1, t2['hsi']), tr.extract_features(m2, t2['lidar'])
    if v1 is not None:
        with torch.no_grad():
            v1.eval(); v2.eval(); ts1 = torch.FloatTensor(f5).to(compute_device); ts2 = torch.FloatTensor(f6).to(compute_device)
            _, _ = v1.encode(ts1), v2.encode(ts2)
    oz1 = tr._get_optimized_latent(v1, v2, ag, torch.FloatTensor(f5).to(compute_device), torch.FloatTensor(f6).to(compute_device), timestep_count)
    oz2 = tr._get_optimized_latent(v1, v2, ag, torch.FloatTensor(f1).to(compute_device), torch.FloatTensor(f2).to(compute_device), timestep_count)
    from sklearn.neighbors import KNeighborsClassifier
    kn = KNeighborsClassifier(n_neighbors=neighbor_count); kn.fit(oz2, t1['labels'].ravel()); pr1 = kn.predict(oz1)
    from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score
    oa1 = accuracy_score(t2['labels'].ravel(), pr1); ka1 = cohen_kappa_score(t2['labels'].ravel(), pr1); pr2 = precision_score(t2['labels'].ravel(), pr1, average='weighted', zero_division=0); re1 = recall_score(t2['labels'].ravel(), pr1, average='weighted', zero_division=0); f1_1 = f1_score(t2['labels'].ravel(), pr1, average='weighted', zero_division=0)
    ul = np.unique(t2['labels'].ravel()); ca = {}
    for lb in ul:
        mk = t2['labels'].ravel() == lb
        if np.sum(mk) > 0:
            ca[int(lb)] = accuracy_score(t2['labels'].ravel()[mk], pr1[mk])

if __name__ == '__main__':
    main()
