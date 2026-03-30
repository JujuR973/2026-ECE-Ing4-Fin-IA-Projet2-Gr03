import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128):
        super(DiffusionModel, self).__init__()
        
        # Couche pour transformer le temps 't' en information compréhensible par l'IA
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Le réseau de neurones principal (Convolutionnel)
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, t):
        # x: (Batch, 64, 1) -> (Batch, 1, 64)
        x = x.transpose(1, 2)
        
        # 1. On traite le temps t
        # On normalise t entre 0 et 1 pour aider l'IA
        t = t.float().view(-1, 1) / 300.0
        t_emb = self.time_mlp(t).unsqueeze(-1) # (Batch, hidden_dim, 1)
        
        # 2. On passe la première couche
        h = self.relu(self.conv1(x))
        
        # 3. ON INJECTE LE TEMPS ICI ! 
        # C'est ça qui va donner du "mouvement" à la courbe
        h = h + t_emb 
        
        h = self.relu(self.conv2(h))
        out = self.conv3(h)
        
        return out.transpose(1, 2)