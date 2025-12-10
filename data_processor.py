"""
数据处理模块
对数据进行标准化处理
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler, RobustScaler
from typing import Tuple, Optional


class DataProcessor:
    """数据处理和标准化类"""
    
    SCALER_METHODS = {
        'scale': StandardScaler,
        'normalize': Normalizer,
        'minmax_scale': MinMaxScaler,
        'maxabs_scale': MaxAbsScaler,
        'robust_scale': RobustScaler
    }
    
    def __init__(self, scaler_method: str = 'minmax_scale'):
        """
        初始化数据处理器
        
        Args:
            scaler_method: 标准化方法 ('scale', 'normalize', 'minmax_scale', 'maxabs_scale', 'robust_scale')
        """
        if scaler_method not in self.SCALER_METHODS:
            raise ValueError(f"不支持的标准化方法: {scaler_method}")
        
        self.scaler_method = scaler_method
        self.hsi_scaler = None
        self.lidar_scaler = None
    
    def normalize_hsi(self, hsi: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        标准化HSI数据
        
        Args:
            hsi: (H, W, B) 的HSI数据
            fit: 是否拟合标准化器
        
        Returns:
            标准化后的 (H, W, B) 的HSI数据
        """
        H, W, B = hsi.shape
        hsi_flat = hsi.reshape(-1, B)
        
        if self.hsi_scaler is None:
            self.hsi_scaler = self.SCALER_METHODS[self.scaler_method]()
        
        if fit:
            hsi_normalized = self.hsi_scaler.fit_transform(hsi_flat)
        else:
            hsi_normalized = self.hsi_scaler.transform(hsi_flat)
        
        hsi_normalized = hsi_normalized.reshape(H, W, B)
        
        return hsi_normalized
    
    def normalize_lidar(self, lidar: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        标准化LiDAR数据
        
        Args:
            lidar: (H, W) 或 (H, W, 1或2) 的LiDAR数据
            fit: 是否拟合标准化器
        
        Returns:
            标准化后的 (H, W, 1或2) 的LiDAR数据
        """
        original_shape = lidar.shape
        
        # 处理2D数据 (H, W)
        if len(lidar.shape) == 2:
            H, W = lidar.shape
            lidar = lidar.reshape(H, W, 1)  # 转换为 (H, W, 1)
            C = 1
        else:
            H, W, C = lidar.shape
        
        lidar_flat = lidar.reshape(-1, C)
        
        if self.lidar_scaler is None:
            self.lidar_scaler = self.SCALER_METHODS[self.scaler_method]()
        
        if fit:
            lidar_normalized = self.lidar_scaler.fit_transform(lidar_flat)
        else:
            lidar_normalized = self.lidar_scaler.transform(lidar_flat)
        
        lidar_normalized = lidar_normalized.reshape(H, W, C)
        
        return lidar_normalized
    
    def process_data(self, hsi: np.ndarray, lidar: np.ndarray, 
                    fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理HSI和LiDAR数据
        
        Args:
            hsi: (H, W, B) 的HSI数据
            lidar: (H, W, 1或2) 的LiDAR数据
            fit: 是否拟合标准化器
        
        Returns:
            标准化后的 (hsi, lidar)
        """
        print(f"使用 {self.scaler_method} 方法进行数据标准化...")
        
        hsi_normalized = self.normalize_hsi(hsi, fit=fit)
        lidar_normalized = self.normalize_lidar(lidar, fit=fit)
        
        print(f"数据标准化完成")
        print(f"HSI数据范围: [{hsi_normalized.min():.4f}, {hsi_normalized.max():.4f}]")
        print(f"LiDAR数据范围: [{lidar_normalized.min():.4f}, {lidar_normalized.max():.4f}]")
        
        return hsi_normalized, lidar_normalized
