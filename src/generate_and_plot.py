import torch
import matplotlib.pyplot as plt
import numpy as np
from src.model import DiffusionModel
from src.diffusion_engine import DiffusionProcess

def generate():
    # 1. Charger le matériel
    model = DiffusionModel()
    model.load_state_dict(torch.load("model_diffusion_E3.pth"))
    model.eval() # On dit à l'IA qu'on n'apprend plus, on utilise
    
    engine = DiffusionProcess(steps=300)
    
    # 2. Créer un bruit de départ (64 jours de pur chaos)
    # Shape: (1 exemplaire, 64 jours, 1 colonne de prix)
    x = torch.randn((1, 64, 1))
    
    print("Génération de la courbe en cours...")
    
    # 3. Le processus inverse (Reverse Diffusion) avec une touche de chaos
    with torch.no_grad():
        for i in reversed(range(engine.steps)):
            t = torch.tensor([i])
            pred_noise = model(x, t)
            
            # On nettoie un peu
            x = x - (0.07 * pred_noise) 
            
            # ASTUCE : On rajoute un tout petit peu de bruit aléatoire 
            # pendant qu'on nettoie pour éviter que la courbe ne devienne plate.
            if i > 0:
                noise = torch.randn_like(x) * 0.01
                x = x + noise

    # 4. Affichage graphique
    plt.figure(figsize=(10, 5))
    plt.plot(x[0].numpy(), label="Donnée Synthétique (IA)", color='blue')
    plt.title("Série temporelle générée par le modèle de Diffusion (E.3)")
    plt.xlabel("Jours")
    plt.ylabel("Rendements (Normalisés)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    generate()