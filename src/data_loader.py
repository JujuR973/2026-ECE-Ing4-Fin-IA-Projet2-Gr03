import yfinance as yf    # Outil pour aller chercher les prix sur Yahoo Finance
import numpy as np       # Outil pour faire des calculs mathématiques sur des listes
import torch             # La bibliothèque d'IA (PyTorch)
from sklearn.preprocessing import StandardScaler # Outil pour mettre les chiffres à la même échelle

def get_financial_data(ticker="^GSPC"):
    # On télécharge les prix de clôture (Close) du S&P 500
    df = yf.download(ticker, start="2015-01-01", end="2025-12-31")
    prices = df['Close']
    
    # En finance, on calcule la variation en % (Log-Returns)
    # car c'est plus stable pour l'IA que le prix en dollars
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    # On transforme ça en une colonne de chiffres propre
    return log_returns.values.reshape(-1, 1)

def prepare_dataloader(data, window_size=64):
    # L'IA a du mal avec les grands chiffres (ex: 4500) et les petits (ex: 0.001)
    # StandardScaler transforme tout pour que la moyenne soit 0 et l'écart-type 1
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # On découpe l'historique en "fenêtres" de 64 jours
    # C'est comme si on donnait des petites photos du marché à l'IA
    sequences = []
    for i in range(len(scaled_data) - window_size):
        sequences.append(scaled_data[i : i + window_size])
    
    # On transforme ces listes en "Tensors" (le format de données spécial pour l'IA)
    dataset = torch.tensor(np.array(sequences), dtype=torch.float32)
    
    # DataLoader permet de donner les données par petits paquets (batchs) à l'IA
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True), scaler