import torch
import matplotlib.pyplot as plt

def generate_new_data(model, engine, shape=(1, 64, 1)):
    model.eval() # Mode "Utilisation"
    with torch.no_grad():
        # On part d'un bruit pur
        x = torch.randn(shape)
        
        # On remonte le temps de T à 0
        for i in reversed(range(engine.steps)):
            t = torch.tensor([i])
            # L'IA prédit le bruit à enlever
            predicted_noise = model(x, t)
            # On retire un peu de bruit
            x = x - (predicted_noise * 0.1) # Version simplifiée du reverse process
            
    return x.numpy()

# Visualisation
# (Ici on afficherait le graphique avec Matplotlib)