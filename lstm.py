import numpy as np
from functions import sigmoid, tanh

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Limite de la distribution uniforme pour Xavier 
        limit = np.sqrt(1 / (input_size + hidden_size))
        
        # Concaténation des dimensions pour W : [h_{t-1}, x_t]
        # La matrice W aura comme dimension : (hidden_size, hidden_size + input_size)
        concat_size = hidden_size + input_size
    
        # 1. Forget Gate (Porte d'oubli)
        self.W_f = np.random.uniform(-limit, limit, (hidden_size, concat_size))
        self.b_f = np.zeros((hidden_size, 1))
        
        # 2. Input Gate (Porte d'entrée)
        self.W_i = np.random.uniform(-limit, limit, (hidden_size, concat_size))
        self.b_i = np.zeros((hidden_size, 1))
        
        # 3. Cell Candidate (Candidat pour l'état de la cellule)
        self.W_c = np.random.uniform(-limit, limit, (hidden_size, concat_size))
        self.b_c = np.zeros((hidden_size, 1))
        
        # 4. Output Gate (Porte de sortie)
        self.W_o = np.random.uniform(-limit, limit, (hidden_size, concat_size))
        self.b_o = np.zeros((hidden_size, 1))
