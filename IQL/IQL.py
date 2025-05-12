import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal

class V_IQL(nn.Module):
    def __init__(self, state_dim):
        super(V_IQL, self).__init__()
        self.state_dim = state_dim 
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, s):
        return self.model(s)

class Q_IQL(nn.Module):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim 
        self.action_dim = action_dim 
        super(Q_IQL, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self._init_weights()
        
    def _init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.bias, 0.1)

        final_layer = self.model[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.uniform_(final_layer.weight, -3e-3, 3e-3)
            nn.init.uniform_(final_layer.bias, -3e-3, 3e-3)

    def forward(self, s, a):
        s = s.float()
        a = a.float()
        x = torch.cat((s, a), dim=1)
        return self.model(x)

class Policy_IQL(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.mean = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, action_dim),
            
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, s, a):
        mean = self.mean(s)
        std = torch.exp(self.log_std.clamp(-5.0, 2.0))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril).log_prob(a).sum(axis=-1, keepdim=True)

    def get_action(self, s):
        mean = self.mean(s)
        std = torch.exp(self.log_std.clamp(-5.0, 2.0))
        scale_tril = torch.diag(std)
        action = MultivariateNormal(mean, scale_tril).sample() if self.training else MultivariateNormal(mean, scale_tril).mean
        return (self.max_action * action).clamp(-self.max_action, self.max_action)
    
class Policy_IQL_Normal(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
        )
        self.mean_head    = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, s, a):
        x = self.model(s)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        assert torch.isfinite(mean).all(),    "NaN in policy mean"
        assert torch.isfinite(log_std).all(), "NaN in policy log_std"
        std = log_std.exp()
        return Normal(mean, std).log_prob(a).sum(dim=-1)

    def get_action(self, s):
        x = self.model(s)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        std = log_std.exp()
        return Normal(mean, std).sample().clamp(-self.max_action, self.max_action)

    
class Policy_IQL_Det(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s) * self.max_action

    def log_prob(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_pred = self.forward(s)
        diff   = a - a_pred               # shape (B,action_dim)
        # negative squared error
        return - (diff.pow(2).sum(dim=-1, keepdim=True))

    @torch.no_grad()
    def get_action(self, s: torch.Tensor) -> torch.Tensor:
        # same as forward
        return self.forward(s).clamp(-self.max_action, self.max_action)