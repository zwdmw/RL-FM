"""
数据加载模块
加载HSI数据、LiDAR数据和标签文件
"""

import numpy as np
from scipy import io
import os
from typing import Dict, Tuple, Optional


class DataLoader:
    """多模态遥感数据加载器"""
    
    def __init__(self, dataset_path: str, dataset_name: str):
        """
        初始化数据加载器
        
        Args:
            dataset_path: 数据集路径
            dataset_name: 数据集名称 ('HS', 'MUUFL', 'Trento')
        """
        self.dataset_path = os.path.join(dataset_path, dataset_name)
        self.dataset_name = dataset_name
        
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"数据集路径不存在: {self.dataset_path}")
    
    def _get_data_from_mat(self, mat_data: dict, possible_keys: list) -> np.ndarray:
        """
        自适应从.mat文件中获取数据
        
        Args:
            mat_data: 加载的.mat文件数据字典
            possible_keys: 可能的键名列表
        
        Returns:
            数据数组
        """
        # 过滤掉MATLAB的元数据键
        data_keys = [key for key in mat_data.keys() if not key.startswith('__')]
        
        # 首先尝试指定的可能键名
        for key in possible_keys:
            if key in mat_data:
                data = mat_data[key]
                if isinstance(data, np.ndarray):
                    return data
        
        # 如果没有找到，尝试其他非元数据键
        for key in data_keys:
            data = mat_data[key]
            if isinstance(data, np.ndarray) and data.size > 0:
                print(f"警告: 使用键 '{key}' 加载数据")
                return data
        
        # 如果还是没找到，打印所有可用的键
        print(f"可用的键: {data_keys}")
        raise ValueError(f"无法找到数据，尝试的键: {possible_keys}")
    
    def load_houston_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载Houston数据集
        
        Returns:
            hsi: (H, W, B) 的HSI数据
            lidar: (H, W, 1或2) 的LiDAR数据
            labels: (H, W) 的标签图
        """
        print(f"加载Houston数据集...")
        
        # 加载HSI数据
        hsi_file = os.path.join(self.dataset_path, 'Houston.mat')
        hsi_data = io.loadmat(hsi_file)
        hsi = self._get_data_from_mat(hsi_data, ['Houston', 'data', 'hsi'])
        
        # 加载LiDAR数据（Houston_LR）
        lidar_file = os.path.join(self.dataset_path, 'Houston_LR.mat')
        lidar_data = io.loadmat(lidar_file)
        lidar = self._get_data_from_mat(lidar_data, ['Houston_LR', 'lidar', 'data'])
        
        # 加载标签数据
        label_file = os.path.join(self.dataset_path, 'Houston_gt.mat')
        label_data = io.loadmat(label_file)
        labels = self._get_data_from_mat(label_data, ['Houston_gt', 'gt', 'labels'])
        
        print(f"HSI形状: {hsi.shape}, LiDAR形状: {lidar.shape}, 标签形状: {labels.shape}")
        
        return hsi, lidar, labels
    
    def load_muufl_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载MUUFL数据集
        
        Returns:
            hsi: (H, W, B) 的HSI数据
            lidar: (H, W, 1或2) 的LiDAR数据
            labels: (H, W) 的标签图
        """
        print(f"加载MUUFL数据集...")
        
        # 加载HSI数据
        hsi_file = os.path.join(self.dataset_path, 'MUF_HSI.mat')
        hsi_data = io.loadmat(hsi_file)
        hsi = self._get_data_from_mat(hsi_data, ['MUF_HSI', 'hsi', 'data'])
        
        # 加载LiDAR数据
        lidar_file = os.path.join(self.dataset_path, 'MUF_LiDAR.mat')
        lidar_data = io.loadmat(lidar_file)
        lidar = self._get_data_from_mat(lidar_data, ['MUF_LiDAR', 'lidar', 'data'])
        
        # 加载标签数据
        label_file = os.path.join(self.dataset_path, 'MUF_gt.mat')
        label_data = io.loadmat(label_file)
        labels = self._get_data_from_mat(label_data, ['MUF_gt', 'gt', 'labels'])
        
        print(f"HSI形状: {hsi.shape}, LiDAR形状: {lidar.shape}, 标签形状: {labels.shape}")
        
        return hsi, lidar, labels
    
    def load_trento_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载Trento数据集
        
        Returns:
            hsi: (H, W, B) 的HSI数据
            lidar: (H, W, 1或2) 的LiDAR数据
            labels: (H, W) 的标签图
        """
        print(f"加载Trento数据集...")
        
        # 加载HSI数据
        hsi_file = os.path.join(self.dataset_path, 'Trento.mat')
        hsi_data = io.loadmat(hsi_file)
        hsi = self._get_data_from_mat(hsi_data, ['Trento', 'hsi', 'data'])
        
        # 加载LiDAR数据
        lidar_file = os.path.join(self.dataset_path, 'LiDAR.mat')
        lidar_data = io.loadmat(lidar_file)
        lidar = self._get_data_from_mat(lidar_data, ['LiDAR', 'lidar', 'data'])
        
        # 加载标签数据
        label_file = os.path.join(self.dataset_path, 'Trento_gt.mat')
        label_data = io.loadmat(label_file)
        labels = self._get_data_from_mat(label_data, ['Trento_gt', 'gt', 'labels'])
        
        print(f"HSI形状: {hsi.shape}, LiDAR形状: {lidar.shape}, 标签形状: {labels.shape}")
        
        return hsi, lidar, labels
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        根据数据集名称加载对应的数据
        
        Returns:
            hsi: (H, W, B) 的HSI数据
            lidar: (H, W, 1或2) 的LiDAR数据
            labels: (H, W) 的标签图
        """
        if self.dataset_name == 'HS':
            return self.load_houston_data()
        elif self.dataset_name == 'MUUFL':
            return self.load_muufl_data()
        elif self.dataset_name == 'Trento':
            return self.load_trento_data()
        else:
            # 尝试自动检测数据集
            return self._auto_detect_and_load()
    
    def _auto_detect_and_load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        自动检测并加载数据集
        
        Returns:
            hsi: (H, W, B) 的HSI数据
            lidar: (H, W, 1或2) 的LiDAR数据
            labels: (H, W) 的标签图
        """
        print(f"自动检测数据集: {self.dataset_name}")
        
        # 获取目录中的所有.mat文件
        mat_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.mat')]
        print(f"找到的.mat文件: {mat_files}")
        
        hsi_file = None
        lidar_file = None
        label_file = None
        
        # 自动识别文件类型
        for file in mat_files:
            file_lower = file.lower()
            if any(keyword in file_lower for keyword in ['hsi', 'hyperspectral', 'spectral']):
                hsi_file = file
            elif any(keyword in file_lower for keyword in ['lidar', 'laser', 'elevation']):
                lidar_file = file
            elif any(keyword in file_lower for keyword in ['gt', 'ground', 'truth', 'label']):
                label_file = file
        
        print(f"识别的文件 - HSI: {hsi_file}, LiDAR: {lidar_file}, 标签: {label_file}")
        
        # 加载数据
        hsi = self._load_mat_file(hsi_file, ['hsi', 'data', 'hyperspectral'])
        lidar = self._load_mat_file(lidar_file, ['lidar', 'data', 'elevation'])
        labels = self._load_mat_file(label_file, ['gt', 'labels', 'ground_truth'])
        
        print(f"数据形状 - HSI: {hsi.shape}, LiDAR: {lidar.shape}, 标签: {labels.shape}")
        
        return hsi, lidar, labels
    
    def _load_mat_file(self, filename: str, possible_keys: list) -> np.ndarray:
        """
        加载单个.mat文件
        
        Args:
            filename: 文件名
            possible_keys: 可能的键名列表
        
        Returns:
            数据数组
        """
        if filename is None:
            raise ValueError("未找到对应的文件")
        
        file_path = os.path.join(self.dataset_path, filename)
        mat_data = io.loadmat(file_path)
        
        return self._get_data_from_mat(mat_data, possible_keys)
