import torch

class DiffusionProcess:
    def __init__(self, steps=300):
        self.steps = steps
        # On définit une montée progressive du bruit (de 0.0001 à 0.02)
        self.betas = torch.linspace(1e-4, 0.02, steps)
        
        # Calculs mathématiques barbares (Alphas) nécessaires pour 
        # pouvoir ajouter du bruit en 1 seule étape au lieu de 300
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x_0, t):
        """
        Prend une donnée nette (x_0) et lui ajoute du bruit 
        correspondant à l'étape 't'.
        """
        # On crée du bruit aléatoire pur
        noise = torch.randn_like(x_0)
        
        # On mélange la donnée et le bruit selon les formules du projet E.3
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # x_t est la version "salie" de la donnée
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        return x_t, noise