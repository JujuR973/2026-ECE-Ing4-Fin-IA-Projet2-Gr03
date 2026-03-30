import torch
import os
# On importe nos propres briques (nos fichiers .py dans le dossier src)
from src.data_loader import get_financial_data, prepare_dataloader
from src.diffusion_engine import DiffusionProcess
from src.model import DiffusionModel

def run_project():
    print("--- LANCEMENT DU PROJET E.3 ---")
    
    # 1. Préparation des dossiers
    if not os.path.exists('data'):
        os.makedirs('data')

    # 2. Récupération des données (S&P 500 par défaut)
    # On transforme les prix en rendements (log-returns)
    raw_data = get_financial_data("^GSPC")
    
    # On prépare le "chargeur" de données pour l'IA
    # loader: les données découpées en fenêtres de 64 jours
    # scaler: l'outil qui a normalisé les données (on en aura besoin à la fin)
    loader, scaler = prepare_dataloader(raw_data, window_size=64)
    
    # 3. Initialisation de l'IA et du moteur de bruit
    engine = DiffusionProcess(steps=300) # 300 étapes de "salissage"
    model = DiffusionModel()             # Notre réseau de neurones
    
    # L'optimiseur : c'est l'outil qui "corrige" les erreurs de l'IA
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Début de l'entraînement...")
    
    # 4. Boucle d'entraînement (On fait 5 tours complets des données)
    for epoch in range(30):
        total_loss = 0
        for batch in loader:
            # On choisit un moment au hasard dans le processus de bruit (entre 0 et 299)
            t = torch.randint(0, engine.steps, (batch.shape[0],))
            
            # On ajoute du bruit à nos vrais prix
            x_noisy, noise_real = engine.add_noise(batch, t)
            
            # L'IA essaie de deviner ce bruit
            noise_pred = model(x_noisy, t)
            
            # On calcule la distance entre la vérité et la devinette
            loss = torch.nn.functional.mse_loss(noise_pred, noise_real)
            
            # On ajuste les neurones pour faire mieux la prochaine fois
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Époque {epoch+1}/30 | Erreur moyenne : {total_loss/len(loader):.4f}")

    print("--- ENTRAÎNEMENT TERMINÉ ---")
    # On sauvegarde le cerveau de l'IA dans un fichier
    torch.save(model.state_dict(), "model_diffusion_E3.pth")
    print("Modèle sauvegardé sous 'model_diffusion_E3.pth'")

if __name__ == "__main__":
    run_project()