import numpy as np

def sigmoid(x):
    """Fonction d'activation sigmoïde utilisée pour les portes du LSTM"""
    x_clipped = np.clip(x, -500, 500) # np.clip évite l'overflow de l'exponentielle
    return 1 / (1 + np.exp(-x_clipped))

def d_sigmoid(x):
    """Dérivée de la fonction sigmoïde pour le backpropagation"""
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """Fonction d'activation tanh utilisée pour les états de la cellule"""
    return np.tanh(x)

def d_tanh(x):
    """Dérivée de la fonction tanh pour le backpropagation"""
    t = tanh(x)
    return 1 - t**2
