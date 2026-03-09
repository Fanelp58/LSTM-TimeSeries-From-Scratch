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


def forward(self, x_t, h_prev, c_prev, use_forget_gate=True):
        """
        Passe avant (Forward Propagation) pour un pas de temps t.
        x_t : Entrée au temps t, dimension (input_size, 1)
        h_prev : État caché précédent h_{t-1}, dimension (hidden_size, 1)
        c_prev : État de la cellule précédent C_{t-1}, dimension (hidden_size, 1)
        use_forget_gate : Paramètre pour l'ablation study
        """
        # 1. Concaténation de l'état caché précédent et de l'entrée
        # np.vstack empile verticalement. Dimension finale : (hidden_size + input_size, 1)
        concat_x = np.vstack((h_prev, x_t))
        
        # 2. Calcul des portes (Gates) avec produit matriciel (np.dot)
        
        # ABLATION STUDY
        if use_forget_gate:
            f_t = sigmoid(np.dot(self.W_f, concat_x) + self.b_f)
        else:
            # Si désactivée, on force la porte à 1 (le modèle n'oublie jamais rien de c_prev)
            # Cela va saturer la mémoire et causer des instabilités à long terme
            # mais c'est exactement ce que nous voulons tester
            f_t = np.ones((self.hidden_size, 1))
            
        # Input Gate et Cell Candidate
        i_t = sigmoid(np.dot(self.W_i, concat_x) + self.b_i)
        c_bar = tanh(np.dot(self.W_c, concat_x) + self.b_c) 
        
        # 3. Mise à jour de l'état de la cellule C_t
        # Utilisation du Produit de Hadamard (*)
        c_t = f_t * c_prev + i_t * c_bar
        
        # 4. Porte de sortie et nouvel état caché h_t
        o_t = sigmoid(np.dot(self.W_o, concat_x) + self.b_o)
        h_t = o_t * tanh(c_t)
        
        # 5. CACHE POUR LA RÉTROPROPAGATION (Backpropagation)
        # On sauvegarde toutes les variables intermédiaires car on en aura 
        # absolument besoin pour calculer les dérivées (Chain Rule) dans le backward()
        cache = (x_t, h_prev, c_prev, concat_x, f_t, i_t, c_bar, c_t, o_t, h_t)
        
        return h_t, c_t, cache