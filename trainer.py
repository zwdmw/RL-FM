import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import os
import glob
from typing import Dict, Tuple, Optional
from feature_extractor import MLPViTHybridFeatureExtractor, FullyConnectedFeatureExtractor, PrototypicalLoss, Reward, VAELoss, VAE_Wrapper
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from scipy.stats import chi2

class Trainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', random_seed=42):
        self.device = device
        self.random_seed = random_seed
        self._srs(random_seed)
    def _srs(self, s):
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s); torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    
    def _cp(self, m):
        return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)
    
    def compute_counterfactual_distribution(self, action_mean, action_var, truncate_std=3.0):
        """
        计算反事实动作分布（截断并反转概率密度）
        
        反事实分布：在截断区间[μ-3σ, μ+3σ]内，原分布的高概率区域变成低概率，低概率区域变成高概率
        
        方法：在截断区间内采样多个点，计算原分布的概率密度，然后反转权重（低密度点获得高权重）
        
        Args:
            action_mean: 原动作均值 (batch_size, action_dim)
            action_var: 原动作方差 (batch_size, action_dim)
            truncate_std: 截断标准差倍数
        
        Returns:
            cf_action: 反事实动作（基于反转概率密度）
        """
        std = torch.sqrt(action_var + 1e-8)
        lower_bound = action_mean - truncate_std * std
        upper_bound = action_mean + truncate_std * std
        
        # 在截断区间内创建采样点网格（不使用蒙特卡洛，而是使用固定点）
        num_samples = 7  # 使用7个采样点
        # 在[-3σ, 3σ]区间内均匀采样
        sample_offsets = torch.linspace(-truncate_std, truncate_std, num_samples, device=action_mean.device, dtype=action_mean.dtype)
        # (num_samples,) -> (num_samples, 1, 1) -> (num_samples, batch_size, action_dim)
        sample_offsets = sample_offsets.view(-1, 1, 1)
        candidates = action_mean.unsqueeze(0) + sample_offsets * std.unsqueeze(0)  # (num_samples, batch_size, action_dim)
        
        # 计算原分布在这些点的概率密度
        dist = torch.distributions.Normal(action_mean.unsqueeze(0), std.unsqueeze(0))
        log_probs = dist.log_prob(candidates)  # (num_samples, batch_size, action_dim)
        probs = torch.exp(log_probs)  # (num_samples, batch_size, action_dim)
        
        # 反转概率密度：高概率变低权重，低概率变高权重
        # 使用 max_prob - prob + epsilon 作为反转后的权重
        max_prob = probs.max(dim=0, keepdim=True)[0]  # (1, batch_size, action_dim)
        inverted_weights = max_prob - probs + 1e-8  # (num_samples, batch_size, action_dim)
        
        # 归一化权重（使得每个维度的权重和为1）
        inverted_weights = inverted_weights / (inverted_weights.sum(dim=0, keepdim=True) + 1e-8)
        
        # 加权平均得到反事实动作（每个维度独立计算）
        cf_action_mean = (candidates * inverted_weights).sum(dim=0)  # (batch_size, action_dim)
        
        return cf_action_mean
    
    def _lbp(self, rd, ds, cs, bfp=None):
        if bfp and os.path.exists(bfp):
            try:
                return np.load(bfp)
            except:
                pass
        if not os.path.exists(rd):
            return None
        sf = os.path.join(rd, f"{ds}_RL_Fusion_predictions.npy")
        if os.path.exists(sf):
            try:
                return np.load(sf)
            except:
                pass
        fs = glob.glob(os.path.join(rd, f"{ds}_*_seed*_*_predictions.npy"))
        if not fs:
            return None
        bf = max(fs, key=os.path.getmtime)
        if f"_seed{cs}_" in bf:
            fwc = [f for f in fs if f"_seed{cs}_" not in f]
            if fwc:
                bf = max(fwc, key=os.path.getmtime)
            else:
                return None
        try:
            return np.load(bf)
        except:
            return None
    
    def _mt(self, yt, yp1, yp2):
        yt, yp1, yp2 = yt.ravel(), yp1.ravel(), yp2.ravel()
        if len(yt) != len(yp1) or len(yt) != len(yp2):
            raise ValueError("预测结果长度不一致")
        ul = np.unique(np.concatenate([yt, yp1, yp2]))
        nc = len(ul)
        if nc == 2:
            a = np.sum((yp2 == ul[0]) & (yp1 == ul[0]))
            b = np.sum((yp2 == ul[0]) & (yp1 == ul[1]))
            c = np.sum((yp2 == ul[1]) & (yp1 == ul[0]))
            if b + c == 0:
                return 0.0, 1.0
            cs = (abs(b - c) - 1) ** 2 / (b + c)
            return cs, 1 - chi2.cdf(cs, df=1)
        else:
            ct = np.zeros((nc, nc), dtype=int)
            lti = {l: i for i, l in enumerate(ul)}
            for i in range(len(yp1)):
                ct[lti[yp2[i]], lti[yp1[i]]] += 1
            ni, nj = np.sum(ct, axis=1), np.sum(ct, axis=0)
            d = ni - nj
            V = np.zeros((nc, nc))
            for i in range(nc):
                for j in range(nc):
                    V[i, j] = (ni[i] + nj[i] - 2 * ct[i, i]) if i == j else -(ct[i, j] + ct[j, i])
            if nc > 1:
                try:
                    Vr, dr = V[:-1, :-1], d[:-1]
                    V_inv = np.linalg.pinv(Vr)
                    cs = np.dot(dr, np.dot(V_inv, dr))
                    return cs, 1 - chi2.cdf(cs, df=nc - 1)
                except:
                    return 0.0, 1.0
            return 0.0, 1.0
    
    def _sm(self, m, sp, md, ep=None):
        os.makedirs(sp, exist_ok=True)
        fn = f"{md}_model.pth" if ep is not None else f"{md}_model_latest.pth"
        torch.save({'model_state_dict': m.state_dict(), 'modality': md, 'epoch': ep}, os.path.join(sp, fn))
    def load_model(self, m, mp):
        c = torch.load(mp, map_location=self.device)
        m.load_state_dict(c['model_state_dict'])
        return m
    
   
    
    def _evaluate_model(self, model: nn.Module, 
                       train_data: Dict, test_data: Dict, 
                       modality: str, knn_neighbors: int) -> Dict:
        """
        评估模型性能
        
        Args:
            model: 特征提取器
            train_data: 训练数据
            test_data: 测试数据
            modality: 模态
            knn_neighbors: KNN邻居数
        
        Returns:
            metrics: {'oa': 总体准确率, 'kappa': Kappa系数, 'class_acc': 每类准确率}
        """
        model.eval()
        
        # 提取训练特征
        train_features = self.extract_features(model, train_data[modality], batch_size=32)
        train_labels = train_data['labels'].ravel()
        
        # 提取测试特征
        test_features = self.extract_features(model, test_data[modality], batch_size=32)
        test_labels = test_data['labels'].ravel()
        
        # 训练KNN分类器
        knn = KNeighborsClassifier(n_neighbors=knn_neighbors)
        knn.fit(train_features, train_labels)
        
        # 预测
        predictions = knn.predict(test_features)
        
        # 计算总体准确率
        oa = accuracy_score(test_labels, predictions)
        
        # 计算Kappa系数
        kappa = cohen_kappa_score(test_labels, predictions)
        
        # 计算每类准确率
        unique_labels = np.unique(test_labels)
        class_acc = {}
        for label in unique_labels:
            mask = test_labels == label
            if np.sum(mask) > 0:
                class_oa = accuracy_score(test_labels[mask], predictions[mask])
                class_acc[int(label)] = class_oa
        
        return {
            'oa': oa,
            'kappa': kappa,
            'class_acc': class_acc
        }
    
    def extract_features(self, model: nn.Module,
                         data: np.ndarray, batch_size: int = 32) -> np.ndarray:
         """
         提取特征
         
         Args:
             model: 特征提取器
             data: (N, p, p, C) 的数据
             batch_size: 批大小
         
         Returns:
             features: (N, embed_dim)
         """
         model.eval()
         
         data_tensor = torch.FloatTensor(data).to(self.device)
         dataset = TensorDataset(data_tensor)
         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
         
         all_features = []
         
         with torch.no_grad():
             for batch_data in dataloader:
                 # 测试时使用池化（FullyConnectedFeatureExtractor的默认行为）
                 # use_pooling=True 是默认参数，所以直接调用即可
                 features = model(batch_data[0])
                 all_features.append(features.cpu().numpy())
         
         features = np.vstack(all_features)
         
         return features
    
    
    
    def fuse_latent_variables(self, hsi_vae: nn.Module, lidar_vae: nn.Module,
                              hsi_features: np.ndarray, lidar_features: np.ndarray,
                              batch_size: int = 32) -> np.ndarray:
        """
        使用高斯混合策略融合两模态潜在变量
        
        Args:
            hsi_vae: HSI VAE
            lidar_vae: LiDAR VAE
            hsi_features: HSI特征 (N, hsi_dim)
            lidar_features: LiDAR特征 (N, lidar_dim)
            batch_size: 批大小
        
        Returns:
            fused_z: 融合的潜在变量 (N, latent_dim)
        """
        hsi_vae.eval()
        lidar_vae.eval()
        
        # 转换为PyTorch张量
        hsi_tensor = torch.FloatTensor(hsi_features).to(self.device)
        lidar_tensor = torch.FloatTensor(lidar_features).to(self.device)
        
        # 创建数据加载器
        dataset = TensorDataset(hsi_tensor, lidar_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_fused_z = []
        
        with torch.no_grad():
            for batch_hsi, batch_lidar in dataloader:
                # 编码得到均值和方差
                # VAE_Wrapper.encode() 返回 mu, var (方差已经在wrapper中处理为正数)
                hsi_mu, hsi_var = hsi_vae.encode(batch_hsi)
                lidar_mu, lidar_var = lidar_vae.encode(batch_lidar)
                
                # 标准高斯混合分布（Gaussian Mixture）
                # p(z) = alpha * N(mu_hsi, var_hsi) + (1-alpha) * N(mu_lidar, var_lidar)
                alpha = 0.9 # 混合权重，可以调整或者根据方差动态计算
                
                # 混合均值：E[z] = alpha * mu_hsi + (1-alpha) * mu_lidar
                fused_mu = alpha * hsi_mu + (1 - alpha) * lidar_mu
                
                # 混合方差：Var[z] = alpha * var_hsi + (1-alpha) * var_lidar + alpha*(1-alpha)*(mu_hsi - mu_lidar)^2
                # 第三项是由于两个分布均值不同而产生的额外方差
                mu_diff_sq = (hsi_mu - lidar_mu) ** 2
                fused_var = alpha * hsi_var + (1 - alpha) * lidar_var + alpha * (1 - alpha) * mu_diff_sq
                
                # 通过重参数化获得融合潜在变量
                fused_z = hsi_vae.reparameterize(fused_mu, fused_var)
                
                all_fused_z.append(fused_z.cpu().numpy())
        
        fused_z = np.vstack(all_fused_z)
        
        return fused_z
    
    def collect_trajectory(self, ppo_agent, state, batch_labels, num_timesteps):
        rewards=[]
        actions=[]
        mus=[]
        vars=[]
        states=[]
        cf_rewards=[]
        rewards_0=Reward()(state, batch_labels)
        for t in range(num_timesteps):
            action_mean, action_var = ppo_agent.policy(state)
            action=torch.distributions.Normal(action_mean, action_var).rsample()
            state_new=state+nn.Tanh()(action)
            reward=(Reward()(state_new, batch_labels)-rewards_0)
            cf_action_mean = self.compute_counterfactual_distribution(action_mean, action_var)
            cf_action = cf_action_mean
            cf_state_new = state+nn.Tanh()(cf_action)
            cf_reward=(Reward()(cf_state_new, batch_labels)-rewards_0)
            rewards.append(reward.unsqueeze(0))
            actions.append(action.unsqueeze(0))
            mus.append(action_mean.unsqueeze(0))
            vars.append(action_var.unsqueeze(0))
            states.append(state_new.unsqueeze(0))
            cf_rewards.append(cf_reward.unsqueeze(0))
            state=state_new
        return torch.cat(rewards,0), torch.cat(actions,0), torch.cat(mus,0), torch.cat(vars,0), torch.cat(states,0), torch.cat(cf_rewards,0)
    def collect_trajectory_test(self, ppo_agent, state,num_timesteps):
       
        actions=[]
        mus=[]
        vars=[]
        states=[]
        for t in range(num_timesteps):
            action_mean, action_var = ppo_agent.policy(state)
            action=torch.distributions.Normal(action_mean, action_var).rsample()
            state_new=state+nn.Tanh()(action)
            
          
            actions.append(action.unsqueeze(0))
            mus.append(action_mean.unsqueeze(0))
            vars.append(action_var.unsqueeze(0))
            states.append(state_new.unsqueeze(0))
            state=state_new
        return torch.cat(actions,0), torch.cat(mus,0), torch.cat(vars,0), torch.cat(states,0)
    def train_rl_policy(self, hsi_model: nn.Module, lidar_model: nn.Module,
                       hsi_vae: nn.Module, lidar_vae: nn.Module,
                       train_data: Dict, test_data: Dict,
                       num_timesteps: int = 5, num_episodes: int = 100,
                       ppo_epochs: int = 3, batch_size: int = 32,
                       learning_rate: float = 1e-4, knn_neighbors: int = 5,
                       save_models: bool = True, model_save_path: str = './saved_models',
                       results_dir: Optional[str] = None, dataset: Optional[str] = None,
                       current_seed: Optional[int] = None,
                       best_file_path: Optional[str] = None) -> Tuple:
        """
        强化学习训练 - 使用PPO优化高斯混合分布
        
        冻结特征提取器和VAE，训练策略网络以在T个时间步内通过调整
        高斯混合分布的均值和方差来优化原型匹配损失
        
        Args:
            hsi_model: 预训练的HSI特征提取器（冻结）
            lidar_model: 预训练的LiDAR特征提取器（冻结）
            hsi_vae: 预训练的HSI VAE（冻结）
            lidar_vae: 预训练的LiDAR VAE（冻结）
            train_data: 训练数据
            test_data: 测试数据
            num_timesteps: 每个episode的时间步数 T
            num_episodes: 训练episodes数
            ppo_epochs: PPO更新的epoch数
            batch_size: 批大小
            learning_rate: 学习率
            knn_neighbors: KNN邻居数
            save_models: 是否保存模型
            model_save_path: 模型保存路径
        
        Returns:
            ppo_agent: 训练完成的PPO agent
            rl_losses: 强化学习损失历史
            rl_eval_history: 评估历史
        """
        from ppo_agent import PPOAgent
        hsi_model.eval(); lidar_model.eval(); hsi_vae.eval(); lidar_vae.eval()
        for param in hsi_model.parameters(): param.requires_grad = False
        for param in lidar_model.parameters(): param.requires_grad = False
        for param in hsi_vae.parameters(): param.requires_grad = False
        for param in lidar_vae.parameters(): param.requires_grad = False
        train_hsi_features = self.extract_features(hsi_model, train_data['hsi'], batch_size=batch_size)
        train_lidar_features = self.extract_features(lidar_model, train_data['lidar'], batch_size=batch_size)
        train_labels = train_data['labels'].ravel()
        test_hsi_features = self.extract_features(hsi_model, test_data['hsi'], batch_size=batch_size)
        test_lidar_features = self.extract_features(lidar_model, test_data['lidar'], batch_size=batch_size)
        test_labels = test_data['labels'].ravel()
        train_hsi_tensor = torch.FloatTensor(train_hsi_features).to(self.device)
        train_lidar_tensor = torch.FloatTensor(train_lidar_features).to(self.device)
        with torch.no_grad():
            _, hsi_mu, _ = hsi_vae(train_hsi_tensor[:1])
        latent_dim = hsi_mu.shape[1]
        ppo_agent = PPOAgent(
            latent_dim=latent_dim,
            hidden_dim=256,
            learning_rate=learning_rate,
            device=self.device,
            clip_ratio=0.2,
            entropy_coef=0.01
        )
        
        # 创建原型匹配损失函数
        proto_loss_fn = PrototypicalLoss()
        
        # 转换为PyTorch张量
        test_hsi_tensor = torch.FloatTensor(test_hsi_features).to(self.device)
        test_lidar_tensor = torch.FloatTensor(test_lidar_features).to(self.device)
        train_labels_tensor = torch.LongTensor(train_labels).to(self.device)
        test_labels_tensor = torch.LongTensor(test_labels).to(self.device)
        
        # 训练数据集
        train_dataset = TensorDataset(train_hsi_tensor, train_lidar_tensor, train_labels_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 损失历史
        rl_losses = {
            'episode_reward': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': []
        }
        rl_eval_history = {'oa': [], 'kappa': [], 'class_acc': [], 'episode': [], 
                           'oa_improvement': [], 'kappa_improvement': []}
        best_oa = 0.0
        
        
        # RL训练循环
        for episode in range(num_episodes):
            episode_rewards = []
            
            # 为每个批次的样本进行轨迹收集
            for batch_hsi, batch_lidar, batch_labels in train_dataloader:
                # 初始化：获取初始的高斯混合分布
                with torch.no_grad():
                    hsi_mu_0, hsi_var_0 = hsi_vae.encode(batch_hsi)
                    lidar_mu_0, lidar_var_0 = lidar_vae.encode(batch_lidar)
                    
                    # 初始高斯混合分布（α=0.5，平衡混合）
                    alpha = 0.9 # 混合权重，可以调整或者根据方差动态计算
                
                # 混合均值：E[z] = alpha * mu_hsi + (1-alpha) * mu_lidar
                    mu_t = alpha * hsi_mu_0 + (1 - alpha) * lidar_mu_0
                
                # 混合方差：Var[z] = alpha * var_hsi + (1-alpha) * var_lidar + alpha*(1-alpha)*(mu_hsi - mu_lidar)^2
                # 第三项是由于两个分布均值不同而产生的额外方差
                    mu_t_diff_sq = (hsi_mu_0 - lidar_mu_0) ** 2
                    var_t = alpha * hsi_var_0 + (1 - alpha) * lidar_var_0 + alpha * (1 - alpha) * mu_t_diff_sq
                
                # 通过重参数化获得融合潜在变量
                    state  = hsi_vae.reparameterize(mu_t, var_t)
                    rewards,actions,mus,vars,states,cf_rewards=self.collect_trajectory(ppo_agent,state,batch_labels,num_timesteps)
                    state_values = ppo_agent.value(states)
                    state_values = state_values.view(rewards.size())
                    
                    # 计算原优势函数
                    gamma = 0.99 # 折扣因子
                    lambda_gae = 0.95  # GAE参数
                    
                    advantages = torch.zeros_like(rewards)
                    returns = torch.zeros_like(rewards)
                    last_gae = 0
                    last_return = 0
                
                    # 从后向前计算原GAE和累积回报
                    for t in reversed(range(rewards.size(0))):
                        if t == rewards.size(0) - 1:
                            next_value = 0  # 最后一步的下一个状态值为0
                        else:
                            next_value = state_values[t + 1]
                        
                        # 计算TD误差
                        delta = rewards[t] + gamma * next_value - state_values[t]
                        
                        # 计算GAE
                        advantages[t] = last_gae = delta + gamma * lambda_gae * last_gae
                        
                        # 计算累积回报（用于价值网络训练）
                        returns[t] =rewards[t] + gamma * last_return
                        last_return = returns[t]
                    
                    # 计算反事实优势函数（使用相同的状态序列）
                    cf_state_values = ppo_agent.value(states)
                    cf_state_values = cf_state_values.view(cf_rewards.size())
                    
                    cf_advantages = torch.zeros_like(cf_rewards)
                    last_gae_cf = 0
                    
                    for t in reversed(range(cf_rewards.size(0))):
                        if t == cf_rewards.size(0) - 1:
                            next_value_cf = 0
                        else:
                            next_value_cf = cf_state_values[t + 1]
                        
                        delta_cf = cf_rewards[t] + gamma * next_value_cf - cf_state_values[t]
                        cf_advantages[t] = last_gae_cf = delta_cf + gamma * lambda_gae * last_gae_cf
                    
                    # 最终优势函数 = 原优势 - 反事实优势
                    final_advantages = advantages - cf_advantages
                    
                    # 标准化最终优势函数
                    final_advantages = (final_advantages - final_advantages.mean()) / (final_advantages.std() + 1e-8)
                    
                  
                    

                

                    action_dists = torch.distributions.Normal(mus, scale=vars)

                
                
                    old_log_probs = action_dists.log_prob(actions)
                for KK in range(10):
                    hsi_mu_0, hsi_var_0 = hsi_vae.encode(batch_hsi)
                    lidar_mu_0, lidar_var_0 = lidar_vae.encode(batch_lidar)
                    
                    # 初始高斯混合分布（α=0.5，平衡混合）
                    alpha = 0.9 # 混合权重，可以调整或者根据方差动态计算
                
                # 混合均值：E[z] = alpha * mu_hsi + (1-alpha) * mu_lidar
                    mu_t = alpha * hsi_mu_0 + (1 - alpha) * lidar_mu_0
                
                # 混合方差：Var[z] = alpha * var_hsi + (1-alpha) * var_lidar + alpha*(1-alpha)*(mu_hsi - mu_lidar)^2
                # 第三项是由于两个分布均值不同而产生的额外方差
                    mu_t_diff_sq = (hsi_mu_0 - lidar_mu_0) ** 2
                    var_t = alpha * hsi_var_0 + (1 - alpha) * lidar_var_0 + alpha * (1 - alpha) * mu_t_diff_sq
                
                # 通过重参数化获得融合潜在变量
                    state  = hsi_vae.reparameterize(mu_t, var_t)
                
                
                    rewards,_,mus,vars,states,_=self.collect_trajectory(ppo_agent,state,batch_labels,num_timesteps)
                
                   



                    

                    action_dists = torch.distributions.Normal(mus, scale=vars)

                    # 确保action是有效的概率分布
                    new_log_probs = action_dists.log_prob(actions)
                    ratio = torch.exp(new_log_probs - old_log_probs)  # (T, 100)
                    clipped_ratio = torch.clamp(ratio, 1. - 0.4, 1. +0.4)

                    surr1 = ratio * final_advantages.unsqueeze(-1).expand_as(ratio)
                    surr2 = clipped_ratio * final_advantages.unsqueeze(-1).expand_as(ratio)
                    policy_loss = -torch.mean(torch.min(surr1, surr2))
                    # 计算价值网络损失
                    state_values = ppo_agent.value(states)
                    state_values = state_values.view(returns.size())
                    value_loss = torch.nn.functional.mse_loss(state_values, returns)
                  
                   
                
                    
                    # 组合所有损失
                    loss = policy_loss+ 0.1*(value_loss) 
                    ppo_agent.policy_optimizer.zero_grad()
                    ppo_agent.value_optimizer.zero_grad()
                    loss.backward()
                    ppo_agent.policy_optimizer.step()
                    ppo_agent.value_optimizer.step()
                    rl_losses['policy_loss'].append(policy_loss.item())
                    rl_losses['value_loss'].append(value_loss.item())
            
            # 每20个episode进行一次完整评估
            if (episode + 1) % 20 == 0 :
                metrics = self._evaluate_rl_policy(
                    hsi_model, lidar_model, hsi_vae, lidar_vae, ppo_agent,
                    train_hsi_features, train_lidar_features, train_labels,
                    test_hsi_features, test_lidar_features, test_labels,
                    knn_neighbors, num_timesteps,
                    results_dir=results_dir, dataset=dataset, current_seed=current_seed,
                    best_file_path=best_file_path
                )
                
                rl_eval_history['episode'].append(episode + 1)
                rl_eval_history['oa'].append(metrics['oa'])
                rl_eval_history['kappa'].append(metrics['kappa'])
                rl_eval_history['class_acc'].append(metrics['class_acc'])
                rl_eval_history['oa_improvement'].append(metrics['oa_improvement'])
                rl_eval_history['kappa_improvement'].append(metrics['kappa_improvement'])
                if metrics['oa'] > best_oa:
                    best_oa = metrics['oa']
                    if save_models:
                        ppo_agent.save(os.path.join(model_save_path, 'ppo_agent.pth'))
                avg_policy_loss = np.mean(rl_losses['policy_loss'][-20:]) if len(rl_losses['policy_loss']) >= 20 else (np.mean(rl_losses['policy_loss']) if rl_losses['policy_loss'] else 0)
                avg_value_loss = np.mean(rl_losses['value_loss'][-20:]) if len(rl_losses['value_loss']) >= 20 else (np.mean(rl_losses['value_loss']) if rl_losses['value_loss'] else 0)
                print(f"Episode {episode + 1}: 损失[策略={avg_policy_loss:.4f}, 价值={avg_value_loss:.4f}], OA={metrics['oa']:.4f}, Kappa={metrics['kappa']:.4f}, AA={metrics['aa']:.4f}, 最佳OA={best_oa:.4f}")
                print(f"  每类准确率: {', '.join([f'类{k}:{v:.4f}' for k, v in sorted(metrics['class_acc'].items())])}")
        
        if save_models:
            ppo_agent.save(os.path.join(model_save_path, 'ppo_agent_final.pth'))
        
        return ppo_agent, rl_losses, rl_eval_history
    
    def _evaluate_rl_policy(self, hsi_model: nn.Module, lidar_model: nn.Module,
                           hsi_vae: nn.Module, lidar_vae: nn.Module, ppo_agent,
                           train_hsi_features: np.ndarray, train_lidar_features: np.ndarray,
                           train_labels: np.ndarray,
                           test_hsi_features: np.ndarray, test_lidar_features: np.ndarray,
                           test_labels: np.ndarray,
                           knn_neighbors: int, num_timesteps: int,
                           results_dir: Optional[str] = None, dataset: Optional[str] = None,
                           current_seed: Optional[int] = None,
                           best_file_path: Optional[str] = None) -> Dict:
        """
        评估RL策略性能
        
        Args:
            hsi_model: HSI特征提取器
            lidar_model: LiDAR特征提取器
            hsi_vae: HSI VAE
            lidar_vae: LiDAR VAE
            ppo_agent: PPO Agent
            train_hsi_features: 训练HSI特征
            train_lidar_features: 训练LiDAR特征
            train_labels: 训练标签
            test_hsi_features: 测试HSI特征
            test_lidar_features: 测试LiDAR特征
            test_labels: 测试标签
            knn_neighbors: KNN邻居数
            num_timesteps: 时间步数
        
        Returns:
            metrics: 评估指标
        """
        hsi_vae.eval()
        lidar_vae.eval()
        ppo_agent.policy.eval()
        ppo_agent.value.eval()
        
        # 转换为张量
        train_hsi_tensor = torch.FloatTensor(train_hsi_features).to(self.device)
        train_lidar_tensor = torch.FloatTensor(train_lidar_features).to(self.device)
        test_hsi_tensor = torch.FloatTensor(test_hsi_features).to(self.device)
        test_lidar_tensor = torch.FloatTensor(test_lidar_features).to(self.device)
        
       
        unique_labels = np.unique(test_labels)
      
        train_optimized_z = self._get_optimized_latent(
            hsi_vae, lidar_vae, ppo_agent,
            train_hsi_tensor, train_lidar_tensor, num_timesteps
        )
        
        test_optimized_z = self._get_optimized_latent(
            hsi_vae, lidar_vae, ppo_agent,
            test_hsi_tensor, test_lidar_tensor, num_timesteps
        )
        
        # 训练KNN分类器并评估优化后的融合
        knn_optimized = KNeighborsClassifier(n_neighbors=knn_neighbors)
        knn_optimized.fit(train_optimized_z, train_labels)
        optimized_predictions = knn_optimized.predict(test_optimized_z)
        
        # 计算优化后的指标
        optimized_oa = accuracy_score(test_labels, optimized_predictions)
        optimized_kappa = cohen_kappa_score(test_labels, optimized_predictions)
        
        # 计算优化后的每类准确率
        optimized_class_acc = {}
        for label in unique_labels:
            mask = test_labels == label
            if np.sum(mask) > 0:
                class_oa = accuracy_score(test_labels[mask], optimized_predictions[mask])
                optimized_class_acc[int(label)] = class_oa
        
        # 3. 计算改进幅度
        oa_improvement = optimized_oa 
        kappa_improvement = optimized_kappa 
        
        # 4. 计算AA（Average Accuracy - 所有类别准确率的平均值）
        aa = np.mean(list(optimized_class_acc.values()))
        
        mcnemar_result = None
        if results_dir and dataset and current_seed is not None:
            best_predictions = self._lbp(results_dir, dataset, current_seed, best_file_path)
            if best_predictions is not None and len(best_predictions) == len(test_labels):
                try:
                    chi2_stat, p_value = self._mt(test_labels, optimized_predictions, best_predictions)
                    best_oa = accuracy_score(test_labels, best_predictions)
                    mcnemar_result = {'chi2_statistic': chi2_stat, 'p_value': p_value, 'best_oa': best_oa, 'current_oa': optimized_oa, 'improvement': optimized_oa - best_oa}
                except:
                    pass
          
        
        return {
            'oa': optimized_oa,
            'kappa': optimized_kappa,
            'class_acc': optimized_class_acc,
            'aa': aa,
            'oa_improvement': oa_improvement,
            'kappa_improvement': kappa_improvement,
            'mcnemar': mcnemar_result
        }
    
    def _get_optimized_latent(self, hsi_vae: nn.Module, lidar_vae: nn.Module,
                             ppo_agent, hsi_features: torch.Tensor,
                             lidar_features: torch.Tensor, num_timesteps: int,
                             batch_size: int = 32) -> np.ndarray:
        """
        通过RL策略获取优化后的潜在变量
        
        Args:
            hsi_vae: HSI VAE
            lidar_vae: LiDAR VAE
            ppo_agent: PPO Agent
            hsi_features: HSI特征张量
            lidar_features: LiDAR特征张量
            num_timesteps: 时间步数
            batch_size: 批大小
        
        Returns:
            optimized_z: 优化后的潜在变量
        """
        hsi_vae.eval()
        lidar_vae.eval()
        ppo_agent.policy.eval()
        
        dataset = TensorDataset(hsi_features, lidar_features)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_optimized_z = []
        
        with torch.no_grad():
            for batch_hsi, batch_lidar in dataloader:
                # 初始高斯混合分布
               hsi_mu_0, hsi_var_0 = hsi_vae.encode(batch_hsi)
               lidar_mu_0, lidar_var_0 = lidar_vae.encode(batch_lidar)
               # 初始高斯混合分布（α=0.5，平衡混合）
               alpha = 0.9  # 混合权重，可以调整或者根据方差动态计算

               # 混合均值：E[z] = alpha * mu_hsi + (1-alpha) * mu_lidar
               mu_t = alpha * hsi_mu_0 + (1 - alpha) * lidar_mu_0

                # 混合方差：Var[z] = alpha * var_hsi + (1-alpha) * var_lidar + alpha*(1-alpha)*(mu_hsi - mu_lidar)^2
                # 第三项是由于两个分布均值不同而产生的额外方差
               mu_t_diff_sq = (hsi_mu_0 - lidar_mu_0) ** 2
               var_t = alpha * hsi_var_0 + (1 - alpha) * lidar_var_0 + alpha * (1 - alpha) * mu_t_diff_sq
                
                # 通过重参数化获得融合潜在变量
               state  = hsi_vae.reparameterize(mu_t, var_t)
               _,_,_,states=self.collect_trajectory_test(ppo_agent,state,num_timesteps)
               all_optimized_z.append(states[-1])
        
        optimized_z = torch.cat(all_optimized_z)
        return optimized_z.cpu().numpy()
