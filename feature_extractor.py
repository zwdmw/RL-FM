"""
特征提取器模块
使用MLPViT混合特征提取器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import Tuple

class Reward(nn.Module):
    """原型匹配损失"""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算原型匹配损失
        
        Args:
            features: (batch, embed_dim)
            labels: (batch,)
        
        Returns:
            loss: 标量损失
        """
        # 计算每个类的原型（类的特征平均值）
        unique_labels = torch.unique(labels)
        prototypes = []
        
        for label in unique_labels:
            class_features = features[labels == label]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # (num_classes, embed_dim)
        
        # 计算特征到原型的相似性
        # 使用余弦相似性
        features_norm = F.normalize(features, p=2, dim=1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
        
        similarities = torch.mm(features_norm, prototypes_norm.T) / self.temperature
        
        # 标签映射
        label_mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}
        targets = torch.tensor([label_mapping[label.item()] for label in labels], 
                              device=labels.device)
        
        # 交叉熵损失
        loss = F.cross_entropy(similarities, targets,reduction='none')
        
        return -loss
class MLPBlock(nn.Module):
    """MLP块"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MLPViTHybridFeatureExtractor(nn.Module):
    """
    混合特征提取器：MLP + ViT
    按照read.md中的要求实现
    """
    def __init__(self, in_channels, patch_size, feature_dim, depth=2, num_heads=4, mlp_ratio=4., dropout=0.1, random_seed=None):
        super().__init__()
        
        # 设置随机种子（如果提供）
        if random_seed is not None:
            self._set_random_seed(random_seed)
        
        # 第一步：MLP映射 - 逐像素处理
        self.pixel_mlp = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, feature_dim),
            nn.Tanh(),
        )
        
        # 第二步：使用ViTFeatureExtractor处理特征
        self.vit_extractor = ViTFeatureExtractor(
            in_channels=feature_dim,
            patch_size=patch_size,
            feature_dim=feature_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # 初始化权重
        self._init_weights()
    
    def _set_random_seed(self, seed: int):
        """设置随机种子以确保权重初始化可重现"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
    def _init_weights(self):
        # 初始化MLP权重
        for m in self.pixel_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # 输入: (N, H, W, D) - 从训练器传递的格式
        N, H, W, D = x.shape
        
        # 第一步：MLP逐像素映射
        # 展平为 (N*H*W, D) 以便逐像素处理
        x = x.reshape(-1, D)  # (N*H*W, D)
        
        # 通过MLP映射到新的特征维度
        x = self.pixel_mlp(x)  # (N*H*W, feature_dim)
        
        # 第二步：重塑回spatial格式并使用ViTFeatureExtractor
        # 重塑为 (N, H, W, feature_dim)
        x = x.reshape(N, H, W, -1)  # (N, H, W, feature_dim)
        
        # 转换为 (N, feature_dim, H, W) 格式以适配ViTFeatureExtractor
        x = x.permute(0, 3, 1, 2)  # (N, feature_dim, H, W)
        
        # 使用ViTFeatureExtractor处理
        x = self.vit_extractor(x)  # (N, feature_dim)
        
        return x


class FullyConnectedFeatureExtractor(nn.Module):
    """
    灵活的全连接层特征提取器
    输入: (B, P, P, D) -> 输出: (B, d1) 测试时 或 (B*P*P, d1) 训练时
    通过全连接层将每个像素映射到d1维
    训练时：返回所有像素点的特征 (B*P*P, d1)
    测试时：全局平均池化后返回 (B, d1)
    
    Args:
        in_channels: 输入特征维度
        feature_dim: 输出特征维度
        hidden_dims: 隐藏层维度列表，例如 [512, 256] 表示两个隐藏层
                     如果为None，则使用默认的 [feature_dim * 2]
        activation: 激活函数类型，支持 'tanh', 'relu', 'gelu', 'leaky_relu', 'elu'
        use_residual: 是否使用残差连接（只在输入输出维度相同时有效）
        use_batch_norm: 是否使用BatchNorm
        dropout: dropout概率
        random_seed: 随机种子
    """
    def __init__(self, 
                 in_channels, 
                 feature_dim, 
                 hidden_dims=None,
                 activation='leaky_relu',
                 use_residual=False,
                 use_batch_norm=True,
                 dropout=0.1, 
                 random_seed=None):
        super().__init__()
        
        # 设置随机种子（如果提供）
        if random_seed is not None:
            self._set_random_seed(random_seed)
        
        # 默认隐藏层配置
        if hidden_dims is None:
            hidden_dims = [feature_dim * 2]
        
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        
        # 获取激活函数
        self.activation_fn = self._get_activation(activation)
        
        # 构建全连接层
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_batch_norm else None
        self.dropout_layers = nn.ModuleList()
        
        # 构建层序列：in_channels -> hidden_dims[0] -> ... -> hidden_dims[-1] -> feature_dim
        dims = [in_channels] + hidden_dims + [feature_dim]
        
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            
            # 添加BatchNorm（在线性层之前）
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(in_dim))
            
            # 添加线性层
            self.fc_layers.append(nn.Linear(in_dim, out_dim))
            
            # 添加Dropout
            self.dropout_layers.append(nn.Dropout(dropout))
        
        # 检查是否可以使用残差连接（需要输入输出维度相同）
        self.can_use_residual = use_residual and (in_channels == feature_dim)
        
        # 如果使用残差但维度不匹配，添加投影层
        if use_residual and in_channels != feature_dim:
            self.residual_projection = nn.Linear(in_channels, feature_dim)
            self.can_use_residual = True
        else:
            self.residual_projection = None
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 初始化权重
        self._init_weights()
    
    def _get_activation(self, activation):
        """获取激活函数"""
        activation_dict = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'sigmoid': nn.Sigmoid(),
            'silu': nn.SiLU(),
        }
        
        if activation.lower() not in activation_dict:
            raise ValueError(f"Unsupported activation: {activation}. "
                           f"Supported: {list(activation_dict.keys())}")
        
        return activation_dict[activation.lower()]
    
    def _set_random_seed(self, seed: int):
        """设置随机种子以确保权重初始化可重现"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
    def _init_weights(self):
        # 初始化全连接层权重
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 初始化投影层权重
        if self.residual_projection is not None:
            nn.init.kaiming_normal_(self.residual_projection.weight)
            if self.residual_projection.bias is not None:
                nn.init.zeros_(self.residual_projection.bias)
    
    def forward(self, x, use_pooling=True):
        """
        前向传播
        
        Args:
            x: 输入 (B, P, P, D)
            use_pooling: 是否使用平均池化
                        True (测试时): 返回 (B, d1)
                        False (训练时): 返回 (B*P*P, d1)
        """
        # 输入: (B, P, P, D)
        B, P, _, D = x.shape
        
        # 重塑为 (B*P*P, D) 以便逐像素处理
        x = x.reshape(-1, D)  # (B*P*P, D)
        
        # 保存输入用于残差连接
        identity = x
        
        # 通过全连接层映射到新的特征维度
        for i in range(len(self.fc_layers)):
            # BatchNorm（如果使用）
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            
            # 线性层
            x = self.fc_layers[i](x)
            
            # 激活函数（最后一层之后不应用激活函数）
            if i < len(self.fc_layers) - 1:
                x = self.activation_fn(x)
            else:
                # 最后一层：应用激活函数
                x = self.activation_fn(x)
            
            # Dropout
            x = self.dropout_layers[i](x)
        
        # 添加残差连接
        if self.can_use_residual:
            if self.residual_projection is not None:
                # 如果维度不匹配，使用投影层
                identity = self.residual_projection(identity)
            x = x + identity
        
        if use_pooling:
            # 测试时：使用平均池化
            # 重塑回spatial格式 (B, P, P, d1)
            x = x.reshape(B, P, P, -1)  # (B, P, P, d1)
            
            # 转换为 (B, d1, P, P) 格式以适配全局平均池化
            x = x.permute(0, 3, 1, 2)  # (B, d1, P, P)
            
            # 全局平均池化得到 (B, d1, 1, 1)
            x = self.global_avg_pool(x)  # (B, d1, 1, 1)
            
            # 展平为 (B, d1)
            x = x.view(B, -1)  # (B, d1)
        else:
            # 训练时：不使用池化，保持 (B*P*P, d1)
            pass
        
        return x

class ViTFeatureExtractor(nn.Module):
    """HSI Vision Transformer特征提取器"""
    def __init__(self, in_channels, patch_size, feature_dim, depth=1, num_heads=4, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, in_channels, feature_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(feature_dim)
        
        # 初始化位置编码
        self._init_weights()

    def _init_weights(self):
        pos_embed = self.pos_embed
        pos_embed.data.normal_(std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

       
        return x[:, 0] 

class PatchEmbed(nn.Module):
    """将输入的HSI patch转换为序列"""
    def __init__(self, patch_size=7, in_channels=103, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, 
                             stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, 1, 1)
        return x.flatten(2).transpose(1, 2)  # (B, 1, embed_dim)

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, embed_dim=512, num_heads=1, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x



class PrototypicalLoss(nn.Module):
    """原型匹配损失"""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算原型匹配损失
        
        Args:
            features: (batch, embed_dim)
            labels: (batch,)
        
        Returns:
            loss: 标量损失
        """
        # 计算每个类的原型（类的特征平均值）
        unique_labels = torch.unique(labels)
        prototypes = []
        
        for label in unique_labels:
            class_features = features[labels == label]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # (num_classes, embed_dim)
        
        # 计算特征到原型的相似性
        # 使用余弦相似性
        features_norm = F.normalize(features, p=2, dim=1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
        
        similarities = torch.mm(features_norm, prototypes_norm.T) / self.temperature
        
        # 标签映射
        label_mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}
        targets = torch.tensor([label_mapping[label.item()] for label in labels], 
                              device=labels.device)
        
        # 交叉熵损失
        loss = F.cross_entropy(nn.Softmax(dim=1)(similarities), targets)
        
        return loss


class VAE(nn.Module):
    """
    变分自编码器 (VAE)
    用于学习潜在表征和特征重建
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list = None, 
                 activation: str = 'leaky_relu', dropout: float = 0.1, random_seed: int = None):
        """
        Args:
            input_dim: 输入特征维度
            latent_dim: 潜在变量维度
            hidden_dims: 隐藏层维度列表，例如 [256, 128]
            activation: 激活函数类型
            dropout: dropout概率
            random_seed: 随机种子
        """
        super().__init__()
        
        if random_seed is not None:
            self._set_random_seed(random_seed)
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # 获取激活函数
        self.activation_fn = self._get_activation(activation)
        
        # 编码器
        encoder_layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            encoder_layers.append(self.activation_fn)
            encoder_layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 潜在空间参数（均值和对数方差）
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # 解码器
        decoder_layers = []
        dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        
        for i in range(len(dims) - 2):
            decoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            decoder_layers.append(self.activation_fn)
            decoder_layers.append(nn.Dropout(dropout))
        
        # 最后一层不加激活函数
        decoder_layers.append(nn.Linear(dims[-2], dims[-1]))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # 初始化权重
        self._init_weights()
    
    def _set_random_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _get_activation(self, activation):
        """获取激活函数"""
        activation_dict = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'sigmoid': nn.Sigmoid(),
            'silu': nn.SiLU(),
        }
        
        if activation.lower() not in activation_dict:
            raise ValueError(f"Unsupported activation: {activation}")
        
        return activation_dict[activation.lower()]
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """
        编码器：将输入映射到潜在空间的均值和对数方差
        
        Args:
            x: (batch, input_dim)
        
        Returns:
            mu: (batch, latent_dim) 均值
            logvar: (batch, latent_dim) 对数方差
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：从N(mu, var)采样
        
        Args:
            mu: (batch, latent_dim) 均值
            logvar: (batch, latent_dim) 对数方差
        
        Returns:
            z: (batch, latent_dim) 采样的潜在变量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        解码器：从潜在变量重建输入
        
        Args:
            z: (batch, latent_dim)
        
        Returns:
            x_recon: (batch, input_dim) 重建的输入
        """
        x_recon = self.decoder(z)
        return x_recon
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch, input_dim)
        
        Returns:
            x_recon: (batch, input_dim) 重建的输入
            mu: (batch, latent_dim) 均值
            logvar: (batch, latent_dim) 对数方差
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


class VAELoss(nn.Module):
    """
    VAE损失函数：重建损失 + KL散度 + 原型匹配损失
    """
    def __init__(self, recon_weight: float = 1.0, kl_weight: float = 0.01, 
                 proto_weight: float = 0.1, proto_temperature: float = 0.1):
        """
        Args:
            recon_weight: 重建损失权重
            kl_weight: KL散度权重
            proto_weight: 原型匹配损失权重
            proto_temperature: 原型匹配温度参数
        """
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.proto_weight = proto_weight
        self.proto_loss = PrototypicalLoss(temperature=proto_temperature)
    
    def forward(self, x, x_recon, mu, logvar, z, labels):
        """
        计算VAE损失
        
        Args:
            x: (batch, input_dim) 原始输入
            x_recon: (batch, input_dim) 重建的输入
            mu: (batch, latent_dim) 均值
            logvar: (batch, latent_dim) 对数方差
            z: (batch, latent_dim) 采样的潜在变量
            labels: (batch,) 标签
        
        Returns:
            total_loss: 总损失
            recon_loss: 重建损失
            kl_loss: KL散度损失
            proto_loss: 原型匹配损失
        """
        # 重建损失（MSE）
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL散度损失
        # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # 平均到每个样本
        
        # 原型匹配损失（在潜在空间中）
        proto_loss_value = self.proto_loss(z, labels)
        
        # 总损失
        total_loss = (self.recon_weight * recon_loss + 
                     self.kl_weight * kl_loss + 
                     self.proto_weight * proto_loss_value)
        
        return total_loss, recon_loss, kl_loss, proto_loss_value


class VAE_Wrapper(nn.Module):
    """
    VAE 包装器，使其接口与 VAE 类兼容
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list = None,
                 random_seed: int = None):
        """
        Args:
            input_dim: 输入特征维度
            latent_dim: 潜在变量维度
            hidden_dims: 隐藏层维度列表（用于编码器和解码器）
            random_seed: 随机种子
        """
        super().__init__()
        
        if random_seed is not None:
            self._set_random_seed(random_seed)
        
        # 设置默认隐藏层
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # 构建编码器层维度列表：input_dim -> hidden_dims -> latent_dim
        e_num = hidden_dims + [latent_dim]
        
        # 构建解码器层维度列表：latent_dim -> hidden_dims_reversed -> input_dim
        d_num = hidden_dims[::-1]
        
        from model import CD_VAE_1
        self.vae = CD_VAE_1(
            dim=input_dim,
            e_num=e_num,
            euc_num=[],
            d_num=d_num
        )
        
        self.latent_dim = latent_dim
    
    def _set_random_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def encode(self, x):
        """
        编码器：将输入映射到潜在空间的均值和方差
        
        Args:
            x: (batch, input_dim)
        
        Returns:
            mu: (batch, latent_dim) 均值
            var: (batch, latent_dim) 方差
        """
        mean_c, var_c, z_c = self.vae(x, label=1)
        var_c = F.softplus(var_c) + 1e-6
        return mean_c, var_c
    
    def reparameterize(self, mu, var):
        """
        重参数化技巧：从N(mu, var)采样
        
        Args:
            mu: (batch, latent_dim) 均值
            var: (batch, latent_dim) 方差
        
        Returns:
            z: (batch, latent_dim) 采样的潜在变量
        """
        z = mu + var
        return z
    
    def decode(self, z):
        """
        解码器：从潜在变量重建输入
        
        Args:
            z: (batch, latent_dim)
        
        Returns:
            x_recon: (batch, input_dim) 重建的输入
        """
        x_recon = self.vae(z, label=2)
        return x_recon
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch, input_dim)
        
        Returns:
            x_recon: (batch, input_dim) 重建的输入
            mu: (batch, latent_dim) 均值
            var: (batch, latent_dim) 方差
        """
        mean_c, var_c, z_c, rec_x = self.vae(x, label=0)
        # 确保方差为正数
        var_c = F.softplus(var_c) + 1e-6
        return rec_x, mean_c, var_c
