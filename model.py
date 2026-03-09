import numpy as np
from lstm import LSTMCell

class LSTMTimeSeriesModel:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Modèle complet de prédiction de séries temporelles.
        """
        self.hidden_size = hidden_size
        
        # 1. Instanciation de notre brique modulaire (Modularité exigée)
        self.lstm_cell = LSTMCell(input_size, hidden_size)
        
        # 2. Couche linéaire de sortie (Dense Layer) : Y = W_y * h_last + b_y
        # Initialisation Xavier (Glorot) adaptée pour la sortie
        limit = np.sqrt(1 / (hidden_size + output_size))
        self.W_y = np.random.uniform(-limit, limit, (output_size, hidden_size))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, X_sequence, use_forget_gate=True):
        """
        Dépliage temporel du LSTM
        X_sequence : Liste des entrées[x_1, x_2, ..., x_T]
        """
        self.caches =[]
        
        # Initialisation de h_0 et C_0 à zéro (vecteurs colonnes)
        h_t = np.zeros((self.hidden_size, 1))
        c_t = np.zeros((self.hidden_size, 1))
        
        # Boucle sur chaque pas de temps t de la séquence
        for x_t in X_sequence:
            # On utilise la cellule LSTM codée "From Scratch"
            h_t, c_t, cache_t = self.lstm_cell.forward(x_t, h_t, c_t, use_forget_gate)
            self.caches.append(cache_t)
            
        # On utilise le dernier état caché h_T pour prédire la valeur future
        # Équation linéaire : Y_pred = W_y * h_T + b_y
        self.last_h = h_t
        y_pred = np.dot(self.W_y, h_t) + self.b_y
        
        return y_pred

    def backward(self, dy_pred):
        """
        Rétropropagation à travers le temps.
        dy_pred : Gradient de la Loss par rapport à la prédiction
        """
        # 1. Gradient de la couche linéaire de sortie
        # dW_y = dy_pred * h_T^T (Transposée de l'activation précédente)
        self.dW_y = np.dot(dy_pred, self.last_h.T)
        self.db_y = dy_pred
        
        # L'erreur qui redescend vers le LSTM (dh_T)
        dh_next = np.dot(self.W_y.T, dy_pred)
        dc_next = np.zeros((self.hidden_size, 1))
        
        # Initialisation des accumulateurs de gradients temporels (matrices de zéros)
        self.dW_f = np.zeros_like(self.lstm_cell.W_f)
        self.dW_i = np.zeros_like(self.lstm_cell.W_i)
        self.dW_c = np.zeros_like(self.lstm_cell.W_c)
        self.dW_o = np.zeros_like(self.lstm_cell.W_o)
        self.db_f = np.zeros_like(self.lstm_cell.b_f)
        self.db_i = np.zeros_like(self.lstm_cell.b_i)
        self.db_c = np.zeros_like(self.lstm_cell.b_c)
        self.db_o = np.zeros_like(self.lstm_cell.b_o)
        
        # 2. Boucle inverse dans le temps (de T à 1)
        for cache_t in reversed(self.caches):
            # Appel du backward de la cellule
            dx_t, dh_next, dc_next = self.lstm_cell.backward(dh_next, dc_next, cache_t)
            
            # Accumulation des gradients (L'erreur totale est la somme des erreurs temporelles)
            self.dW_f += self.lstm_cell.dW_f
            self.dW_i += self.lstm_cell.dW_i
            self.dW_c += self.lstm_cell.dW_c
            self.dW_o += self.lstm_cell.dW_o
            self.db_f += self.lstm_cell.b_f
            self.db_i += self.lstm_cell.b_i
            self.db_c += self.lstm_cell.b_c
            self.db_o += self.lstm_cell.b_o

    def update_params(self, learning_rate):
        """
        Mise à jour des poids avec SGD (Stochastic Gradient Descent)
        """
        # Mise à jour de la couche de sortie
        self.W_y -= learning_rate * self.dW_y
        self.b_y -= learning_rate * self.db_y
        
        # Mise à jour des poids de la cellule LSTM
        self.lstm_cell.W_f -= learning_rate * self.dW_f
        self.lstm_cell.W_i -= learning_rate * self.dW_i
        self.lstm_cell.W_c -= learning_rate * self.dW_c
        self.lstm_cell.W_o -= learning_rate * self.dW_o
        
        self.lstm_cell.b_f -= learning_rate * self.db_f
        self.lstm_cell.b_i -= learning_rate * self.db_i
        self.lstm_cell.b_c -= learning_rate * self.db_c
        self.lstm_cell.b_o -= learning_rate * self.db_o