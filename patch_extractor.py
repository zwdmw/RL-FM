"""
Patch切割模块
从HSI和LiDAR数据中切割patches
"""

import numpy as np
from typing import Tuple, List


class PatchExtractor:
    """Patch切割类"""
    
    def __init__(self, patch_size: int = 5, background_label: int = None):
        """
        初始化Patch切割器
        
        Args:
            patch_size: patch大小 (p x p)
            background_label: 背景标签值 (默认None，会自动检测最小值)
        """
        self.patch_size = patch_size
        self.background_label = background_label
        self.pad_size = patch_size // 2
    
    def mirror_pad(self, data: np.ndarray, pad_size: int) -> np.ndarray:
        """
        对数据进行镜像填充
        
        Args:
            data: 输入数据 (H, W, C)
            pad_size: 填充大小
        
        Returns:
            填充后的数据 (H+2*pad_size, W+2*pad_size, C)
        """
        H, W = data.shape[:2]
        
        # 对高度进行镜像填充
        if pad_size > 0:
            # 上下填充
            top = data[1:pad_size+1, :, :][::-1]  # 镜像上
            bottom = data[H-pad_size-1:H-1, :, :][::-1]  # 镜像下
            data = np.vstack([top, data, bottom])
            
            # 左右填充
            left = data[:, 1:pad_size+1, :][:, ::-1, :]  # 镜像左
            right = data[:, -pad_size-1:-1, :][:, ::-1, :]  # 镜像右
            data = np.hstack([left, data, right])
        
        return data
    
    def extract_patches(self, hsi: np.ndarray, lidar: np.ndarray, 
                       labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        提取patches
        
        Args:
            hsi: (H, W, B) 的HSI数据
            lidar: (H, W, 1或2) 的LiDAR数据
            labels: (H, W) 的标签图
        
        Returns:
            hsi_patches: (N, p, p, B)
            lidar_patches: (N, p, p, C)
            patch_labels: (N, 1)
        """
        print(f"开始提取patches (patch大小: {self.patch_size}x{self.patch_size})...")
        
        H, W = labels.shape
        
        # 如果未指定背景标签，自动检测最小值
        if self.background_label is None:
            self.background_label = self.get_background_label_value(labels)
            print(f"自动检测背景标签值: {self.background_label}")
        
        # 获取非背景点的索引
        non_bg_mask = labels != self.background_label
        non_bg_indices = np.where(non_bg_mask)
        num_samples = len(non_bg_indices[0])
        
        print(f"背景标签值: {self.background_label}")
        print(f"总像素点数: {H*W}, 非背景点数: {num_samples}, 背景点数: {H*W - num_samples}")
        
        # 对数据进行镜像填充
        hsi_padded = self.mirror_pad(hsi, self.pad_size)
        lidar_padded = self.mirror_pad(lidar, self.pad_size)
        
        H_pad = hsi_padded.shape[0]
        W_pad = hsi_padded.shape[1]
        
        # 初始化patch数组
        B = hsi.shape[2]
        C = lidar.shape[2]
        
        hsi_patches = np.zeros((num_samples, self.patch_size, self.patch_size, B), dtype=np.float32)
        lidar_patches = np.zeros((num_samples, self.patch_size, self.patch_size, C), dtype=np.float32)
        patch_labels = np.zeros((num_samples, 1), dtype=np.int64)
        
        # 对每个非背景点提取patch
        for idx, (h, w) in enumerate(zip(non_bg_indices[0], non_bg_indices[1])):
            # 考虑填充
            h_padded = h + self.pad_size
            w_padded = w + self.pad_size
            
            # 提取patch
            h_start = h_padded - self.pad_size
            h_end = h_padded + self.pad_size + 1
            w_start = w_padded - self.pad_size
            w_end = w_padded + self.pad_size + 1
            
            hsi_patches[idx] = hsi_padded[h_start:h_end, w_start:w_end, :]
            lidar_patches[idx] = lidar_padded[h_start:h_end, w_start:w_end, :]
            patch_labels[idx, 0] = labels[h, w]
            
            if (idx + 1) % 5000 == 0:
                print(f"已提取 {idx + 1}/{num_samples} 个patches")
        
        print(f"Patch提取完成")
        print(f"HSI patches形状: {hsi_patches.shape}")
        print(f"LiDAR patches形状: {lidar_patches.shape}")
        print(f"标签形状: {patch_labels.shape}")
        print(f"标签类别数: {len(np.unique(patch_labels))}")
        
        return hsi_patches, lidar_patches, patch_labels
    
    def get_background_label_value(self, labels: np.ndarray) -> int:
        """
        自动获取背景标签值（标签值最小的）
        
        Args:
            labels: (H, W) 的标签图
        
        Returns:
            背景标签值
        """
        unique_labels = np.unique(labels)
        return int(unique_labels[0])
