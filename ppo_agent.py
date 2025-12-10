import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class _N1(nn.Module):
    def __init__(self, d1, d2=256, d3=None):
        super(_N1, self).__init__()
        self.d1 = d1
        self.d3 = d3 if d3 is not None else 2 * d1
        self.l1 = nn.Linear(d2, d2)
        self.l2 = nn.Linear(d2, d2)
        self.l3 = nn.Linear(d2, d2)
        self.l4 = nn.Linear(d2, d2)
        for m in [self.l1, self.l2, self.l3, self.l4]:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        m1 = torch.tanh(self.l3(x))
        m2 = torch.nn.Softplus()(self.l4(x))
        return m1, m2

class _N2(nn.Module):
    def __init__(self, d1, d2=256):
        super(_N2, self).__init__()
        self.d1 = d1
        self.l1 = nn.Linear(d1, d2)
        self.l2 = nn.Linear(d2, d2)
        self.l3 = nn.Linear(d2, 1)
        for m in [self.l1, self.l2, self.l3]:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        return self.l3(x)

class PPOAgent:
    def __init__(self, latent_dim, hidden_dim=256, learning_rate=1e-4, device='cuda', clip_ratio=0.2, entropy_coef=0.01):
        self.latent_dim = latent_dim
        self.device = device
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.policy = _N1(latent_dim*2, hidden_dim).to(device)
        self.value = _N2(latent_dim, hidden_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=learning_rate)
        self.memory = {'states': [], 'actions': [], 'rewards': [], 'values': [], 'log_probs': []}
    def select_action(self, s):
        if isinstance(s, torch.Tensor):
            st = s.unsqueeze(0) if s.dim() == 1 else s
            if st.device.type != self.device.split(':')[0]:
                st = st.to(self.device)
        else:
            st = torch.FloatTensor(s).unsqueeze(0).to(self.device)
        with torch.no_grad():
            m1, m2 = self.policy(st)
            n1, n2 = m1, m2
            lp = -0.5 * (n1 ** 2 + n2 ** 2).sum()
        return (m1 + n1).cpu().numpy()[0], (m2 + n2).cpu().numpy()[0], lp.item()
    def store_transition(self, s, a, r, v, lp):
        self.memory['states'].append(s)
        self.memory['actions'].append(a)
        self.memory['rewards'].append(r)
        self.memory['values'].append(v)
        self.memory['log_probs'].append(lp)
    def compute_returns_and_advantages(self, g=0.99, gl=0.95):
        r = np.array(self.memory['rewards'])
        v = np.array(self.memory['values'])
        td = r + g * np.append(v[1:], 0) - v
        adv = np.zeros_like(td)
        gae = 0
        for t in reversed(range(len(td))):
            gae = td[t] + g * gl * gae
            adv[t] = gae
        ret = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return ret, adv
    def update(self, ne=3, bs=32):
        if len(self.memory['states']) == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0}
        ret, adv = self.compute_returns_and_advantages()
        sl = []
        for s in self.memory['states']:
            sl.append(s.cpu().detach().numpy() if isinstance(s, torch.Tensor) else s)
        st = torch.FloatTensor(np.array(sl)).to(self.device)
        olp = torch.clamp(torch.FloatTensor(self.memory['log_probs']).to(self.device), -20, 20)
        ret = torch.FloatTensor(ret).to(self.device)
        adv = torch.FloatTensor(adv).to(self.device)
        ld = {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0}
        nb = 0
        for e in range(ne):
            idx = np.random.permutation(len(st))
            for i in range(0, len(st), bs):
                bi = idx[i:i + bs]
                bst = st[bi]
                bolp = olp[bi]
                bret = ret[bi]
                badv = adv[bi]
                m1, m2 = self.policy(bst)
                nlp = torch.clamp(-0.5 * (m1 ** 2 + m2 ** 2).sum(dim=1), -20, 20)
                lr = torch.clamp(nlp - bolp, -20, 20)
                r = torch.clamp(torch.exp(lr), 0, 1e6)
                s1 = r * badv
                s2 = torch.clamp(r, 1 - self.clip_ratio, 1 + self.clip_ratio) * badv
                pl = -torch.min(s1, s2).mean()
                ent = torch.clamp(0.5 * (1 + torch.log(2 * np.pi * (m1.std(dim=1) ** 2 + 1e-8))).mean(), -10, 10)
                tpl = pl - self.entropy_coef * ent
                vl = torch.clamp(F.mse_loss(self.value(bst).squeeze(), bret), 0, 1e6)
                self.policy_optimizer.zero_grad()
                tpl.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()
                self.value_optimizer.zero_grad()
                vl.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.value_optimizer.step()
                ld['policy_loss'] += pl.item()
                ld['value_loss'] += vl.item()
                ld['entropy_loss'] += ent.item()
                nb += 1
        if nb > 0:
            for k in ld:
                ld[k] /= nb
        self.clear_memory()
        return ld
    def clear_memory(self):
        self.memory = {'states': [], 'actions': [], 'rewards': [], 'values': [], 'log_probs': []}
    def save(self, p):
        torch.save({'policy_state_dict': self.policy.state_dict(), 'value_state_dict': self.value.state_dict()}, p)
    def load(self, p):
        c = torch.load(p, map_location=self.device)
        self.policy.load_state_dict(c['policy_state_dict'])
        self.value.load_state_dict(c['value_state_dict'])
