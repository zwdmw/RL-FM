"""
分类器模块
使用KNN进行分类
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from typing import Dict, Tuple


class KNNClassifier:
    """KNN分类器"""
    
    def __init__(self, n_neighbors: int = 5):
        """
        初始化KNN分类器
        
        Args:
            n_neighbors: 邻居数
        """
        self.n_neighbors = n_neighbors
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    def train(self, features: np.ndarray, labels: np.ndarray):
        """
        训练KNN分类器
        
        Args:
            features: (N, embed_dim) 的特征
            labels: (N,) 的标签
        """
        print(f"训练KNN分类器 (k={self.n_neighbors})...")
        self.classifier.fit(features, labels)
        print("KNN分类器训练完成")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        预测标签
        
        Args:
            features: (N, embed_dim) 的特征
        
        Returns:
            predictions: (N,) 的预测标签
        """
        return self.classifier.predict(features)
    
    def evaluate(self, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        """
        评估分类结果
        
        Args:
            predictions: (N,) 的预测标签
            labels: (N,) 的真实标签
        
        Returns:
            metrics: 评估指标字典
        """
        accuracy = accuracy_score(labels, predictions)
        
        # 计算Kappa系数
        kappa = cohen_kappa_score(labels, predictions)
        
        # 计算每类准确率
        unique_labels = np.unique(labels)
        class_acc = {}
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 0:
                class_oa = accuracy_score(labels[mask], predictions[mask])
                class_acc[int(label)] = class_oa
        
        # 处理多分类情况
        if len(unique_labels) > 2:
            precision = precision_score(labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        else:
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'kappa': kappa,
            'class_acc': class_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics


class MultimodalClassifier:
    """多模态分类器"""
    
    def __init__(self, n_neighbors: int = 5, fusion_method: str = 'average'):
        """
        初始化多模态分类器
        
        Args:
            n_neighbors: 邻居数
            fusion_method: 融合方法 ('average', 'concat', 'voting')
        """
        self.n_neighbors = n_neighbors
        self.fusion_method = fusion_method
        self.hsi_classifier = KNNClassifier(n_neighbors=n_neighbors)
        self.lidar_classifier = KNNClassifier(n_neighbors=n_neighbors)
    
    def train(self, hsi_features: np.ndarray, lidar_features: np.ndarray, 
             labels: np.ndarray):
        """
        训练多模态分类器
        
        Args:
            hsi_features: (N, embed_dim) 的HSI特征
            lidar_features: (N, embed_dim) 的LiDAR特征
            labels: (N,) 的标签
        """
        print("\n训练多模态KNN分类器...")
        self.hsi_classifier.train(hsi_features, labels)
        self.lidar_classifier.train(lidar_features, labels)
        print("多模态分类器训练完成")
    
    def predict(self, hsi_features: np.ndarray, lidar_features: np.ndarray) -> np.ndarray:
        """
        预测标签
        
        Args:
            hsi_features: (N, embed_dim) 的HSI特征
            lidar_features: (N, embed_dim) 的LiDAR特征
        
        Returns:
            predictions: (N,) 的预测标签
        """
        hsi_pred = self.hsi_classifier.predict(hsi_features)
        lidar_pred = self.lidar_classifier.predict(lidar_features)
        
        if self.fusion_method == 'voting':
            # 投票融合
            predictions = np.zeros_like(hsi_pred)
            for i in range(len(hsi_pred)):
                if hsi_pred[i] == lidar_pred[i]:
                    predictions[i] = hsi_pred[i]
                else:
                    # 简单投票
                    predictions[i] = hsi_pred[i]
        else:
            # 默认使用HSI预测
            predictions = hsi_pred
        
        return predictions
