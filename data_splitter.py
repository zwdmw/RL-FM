"""
训练/测试集划分模块
"""

import numpy as np
from typing import Tuple


class DataSplitter:
    """数据集划分类"""
    
    def __init__(self, random_seed: int = 42, background_label: int = None):
        """
        初始化数据集划分器
        
        Args:
            random_seed: 随机种子
            background_label: 背景标签值（默认None，会自动检测最小值）
        """
        self.random_seed = random_seed
        self.background_label = background_label
        np.random.seed(random_seed)
    
    def split_data(self, hsi_patches: np.ndarray, lidar_patches: np.ndarray, 
                   labels: np.ndarray, samples_per_class: int) -> Tuple[dict, dict]:
        """
        根据标签将数据分成训练集和测试集
        每类随机选择Q个样本作为训练集，其余作为测试集
        
        Args:
            hsi_patches: (N, p, p, B) 的HSI patches
            lidar_patches: (N, p, p, C) 的LiDAR patches
            labels: (N, 1) 的标签
            samples_per_class: 每类训练样本数 Q
        
        Returns:
            train_data: {'hsi': (N_train, p, p, B), 'lidar': (N_train, p, p, C), 'labels': (N_train, 1)}
            test_data: {'hsi': (N_test, p, p, B), 'lidar': (N_test, p, p, C), 'labels': (N_test, 1)}
        """
        print(f"数据集划分: 每类 {samples_per_class} 个训练样本...")
        
        labels_flat = labels.ravel()
        unique_labels = np.unique(labels_flat)
        
        # 自动检测背景标签值（最小值）
        if self.background_label is None:
            self.background_label = int(unique_labels[0])
            print(f"自动检测背景标签值: {self.background_label}")
        
        # 排除背景标签
        unique_labels = unique_labels[unique_labels != self.background_label]
        print(f"排除背景标签 {self.background_label}，剩余类别: {len(unique_labels)}")
        
        train_indices = []
        test_indices = []
        
        for label in unique_labels:
            label_indices = np.where(labels_flat == label)[0]
            num_samples = len(label_indices)
            
            # 随机打乱索引
            shuffled_indices = np.random.permutation(label_indices)
            
            # 选择前Q个作为训练集
            train_idx = shuffled_indices[:min(samples_per_class, num_samples)]
            test_idx = shuffled_indices[min(samples_per_class, num_samples):]
            
            train_indices.extend(train_idx)
            test_indices.extend(test_idx)
            
            print(f"类别 {label}: 总样本数 {num_samples}, 训练集 {len(train_idx)}, 测试集 {len(test_idx)}")
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
        # 创建训练集
        train_data = {
            'hsi': hsi_patches[train_indices],
            'lidar': lidar_patches[train_indices],
            'labels': labels[train_indices]
        }
        
        # 创建测试集
        test_data = {
            'hsi': hsi_patches[test_indices],
            'lidar': lidar_patches[test_indices],
            'labels': labels[test_indices]
        }
        
        print(f"\n数据集划分完成:")
        print(f"训练集大小: {len(train_indices)}")
        print(f"测试集大小: {len(test_indices)}")
        print(f"总样本数: {len(train_indices) + len(test_indices)}")
        
        return train_data, test_data
