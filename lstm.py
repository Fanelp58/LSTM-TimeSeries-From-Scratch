import numpy as np
from functions import sigmoid, tanh

class LSTMCell:
    """
    Définition modulaire de la cellule LSTM
    """
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
        self.b_f = np.ones((hidden_size, 1)) # ANTI-VANISHING GRADIENT : Initialisation de b_f à 1
        
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
        Passe avant (Forward Propagation) pour un pas de temps t
        x_t : Entrée au temps t, dimension (input_size, 1)
        h_prev : État caché précédent h_{t-1}, dimension (hidden_size, 1)
        c_prev : État de la cellule précédent C_{t-1}, dimension (hidden_size, 1)
        use_forget_gate : Paramètre pour l'ablation study
        """
        # 1. Concaténation de l'état caché précédent et de l'entrée
        # np.vstack empile verticalement. Dimension finale : (hidden_size + input_size, 1)
        concat_x = np.vstack((h_prev, x_t))
                
        # ABLATION STUDY
        if use_forget_gate:
            f_t = sigmoid(np.dot(self.W_f, concat_x) + self.b_f)
        else:
            # Si désactivée, on force la porte à 1 (le modèle n'oublie jamais rien de c_prev)
            # Cela peut saturer la mémoire et causer des instabilités à long terme
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
        
        # 5. CACHE POUR LA BACKPROPAGATION
        # On sauvegarde toutes les variables intermédiaires car on en aura 
        # besoin pour calculer les dérivées (Chain rule) dans le backward()
        cache = (x_t, h_prev, c_prev, concat_x, f_t, i_t, c_bar, c_t, o_t, h_t)
        
        return h_t, c_t, cache


    def backward(self, dh_next, dc_next, cache):
        """
        Rétropropagation pour un pas de temps t
        dh_next : Gradient de la loss par rapport à h_t, dimension (hidden_size, 1)
        dc_next : Gradient de la loss par rapport à c_t, dimension (hidden_size, 1)
        """
        # 1. Récupération
        x_t, h_prev, c_prev, concat_x, f_t, i_t, c_bar, c_t, o_t, h_t = cache
        
        # 2. Gradient de l'état de la cellule (c_t)
        dc_t = dc_next + (dh_next * o_t * (1 - np.square(np.tanh(c_t))))
        
        # 3. Gradients des portes (Hadammard product)
        do_t = dh_next * np.tanh(c_t)
        df_t = dc_t * c_prev
        di_t = dc_t * c_bar
        dc_bar = dc_t * i_t
        
        # 4. Dérivées locales (C'est ici que s'applique l'ablation)
        dZ_o = do_t * o_t * (1 - o_t)
        dZ_i = di_t * i_t * (1 - i_t)
        dZ_c = dc_bar * (1 - np.square(c_bar))
        
        # APPLICATION CONDITIONNELLE : Si f_t est constant à 1, pas de gradient pour la Forget Gate (dérivée nulle)
        if np.all(f_t == 0) or np.all(f_t == 1):
            dZ_f = np.zeros_like(f_t)
        else:
            dZ_f = df_t * f_t * (1 - f_t)
                
        # 5. Calcul des gradients des poids (dJ/dW)
        # dW = dZ * A_prev.T (Transposée de l'entrée)
        # dimensions: (hidden_size, 1) dot (1, hidden_size + input_size) => (hidden_size, hidden_size + input_size)
        self.dW_f = np.dot(dZ_f, concat_x.T)  
        self.dW_i = np.dot(dZ_i, concat_x.T)
        self.dW_c = np.dot(dZ_c, concat_x.T)
        self.dW_o = np.dot(dZ_o, concat_x.T)

        # Calcul des gradients des biais (db)
        self.db_f = np.sum(dZ_f, axis=1, keepdims=True)
        self.db_i = np.sum(dZ_i, axis=1, keepdims=True)
        self.db_c = np.sum(dZ_c, axis=1, keepdims=True)
        self.db_o = np.sum(dZ_o, axis=1, keepdims=True)
        
        # 6. Propagation de l'erreur vers l'entrée et le passé (dx_t, dh_prev)
        # dZ_prev = W.T * dZ (On utilise W Transposé)
        d_concat_x = (np.dot(self.W_f.T, dZ_f) + 
                      np.dot(self.W_i.T, dZ_i) + 
                      np.dot(self.W_c.T, dZ_c) + 
                      np.dot(self.W_o.T, dZ_o))
        
        # On sépare le gradient concaténé pour retrouver dh_prev et dx_t
        dh_prev = d_concat_x[:self.hidden_size, :]
        dx_t = d_concat_x[self.hidden_size:, :]
        
        # Gradient de l'état de la cellule précédent
        dc_prev = dc_t * f_t
        
        return dx_t, dh_prev, dc_prev

