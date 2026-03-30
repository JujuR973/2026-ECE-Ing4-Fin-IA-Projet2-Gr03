import torch
from data_loader import get_financial_data, prepare_dataloader
from diffusion_engine import DiffusionProcess
from model import DiffusionModel

# 1. Préparation
data = get_financial_data("^GSPC")
loader, scaler = prepare_dataloader(data)
engine = DiffusionProcess(steps=300)
model = DiffusionModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # L'algorithme qui corrige l'IA

# 2. Boucle d'apprentissage
for epoch in range(10): # On passe 10 fois sur toutes les données
    for batch in loader:
        # On choisit une étape de bruit au hasard pour chaque donnée
        t = torch.randint(0, 300, (batch.shape[0],))
        
        # On "salit" la donnée
        x_noisy, noise_real = engine.add_noise(batch, t)
        
        # L'IA essaie de deviner le bruit
        noise_pred = model(x_noisy, t)
        
        # On calcule l'erreur (Loss) entre le vrai bruit et la prédiction
        loss = torch.nn.functional.mse_loss(noise_pred, noise_real)
        
        # Correction de l'IA
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch} | Erreur: {loss.item():.4f}")