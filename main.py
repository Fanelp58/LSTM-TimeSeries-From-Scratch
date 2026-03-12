import numpy as np
import matplotlib.pyplot as plt
from model import LSTMTimeSeriesModel


# 1. GÉNÉRATION DU DATASET (Série temporelle)
def generate_data(seq_length, num_samples):
    """
    Génère une courbe sinusoïdale continue.
    """
    x = np.linspace(0, 50, num_samples)
    y = np.sin(x)
    X, Y = [], []
    for i in range(len(y) - seq_length):
        # Format colonne indispensable pour les exigences matricielles : (input_size, 1)
        seq = y[i:i + seq_length].reshape(-1, 1)
        target = y[i + seq_length].reshape(1, 1)
        X.append(seq)
        Y.append(target)
    return X, Y

# 2. FONCTION D'ENTRAÎNEMENT
def train_model(X, Y, epochs, learning_rate, use_forget_gate=True):
    model = LSTMTimeSeriesModel(input_size=1, hidden_size=7, output_size=1)
    losses =[]
    
    print(f"Début de l'entraînement (Forget Gate = {use_forget_gate}) ")
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(X)):
            x_seq = X[i]
            y_true = Y[i]
            
            # 1. Forward Pass (Prédiction)
            y_pred = model.forward(x_seq, use_forget_gate)
            
            # 2. Calcul de la Loss (Mean Squared Error - MSE)
            loss = 0.5 * np.square(y_pred - y_true)
            epoch_loss += loss[0, 0]
            
            # 3. Gradient de la Loss (dérivée de 0.5 * (y_pred - y_true)^2)
            # Le gradient à la sortie est simplement la différence entre la prédiction et la valeur réelle
            #  : A[L] - Y
            dy_pred = y_pred - y_true
            
            # 4. Backward Pass et Mise à jour des poids (SGD)
            model.backward(dy_pred)
            model.update_params(learning_rate)
            
        avg_loss = epoch_loss / len(X)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")
            
    return model, losses

# 3. EXÉCUTION ET ABLATION STUDY 
if __name__ == "__main__":
    np.random.seed(42)

    # Paramètres
    SEQ_LENGTH = 50
    NUM_SAMPLES = 500
    EPOCHS = 50
    LR = 0.01
    
    # Préparation des données
    X, Y = generate_data(SEQ_LENGTH, NUM_SAMPLES)
    
    # Entraînement 1 : Modèle LSTM complet (normal)
    model_complet, loss_complet = train_model(X, Y, EPOCHS, LR, use_forget_gate=True)
    
    # Entraînement 2 : Modèle LSTM sans forget gate (Ablation study)
    model_ablation, loss_ablation = train_model(X, Y, EPOCHS, LR, use_forget_gate=False)
    
    # 4. AFFICHAGE DES RÉSULTATS (Pour le PDF)
    plt.figure(figsize=(10, 5))
    plt.plot(loss_complet, label="LSTM Complet (Avec forget gate)", color="blue", linewidth=2)
    plt.plot(loss_ablation, label="LSTM Modifié (Sans forget gate)", color="red", linestyle="dashed", linewidth=2)
    plt.title("Ablation study : impact de la forget gate sur la convergence")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    
    
    plt.savefig("ablation_study_results.png")
    print("Graphique sauvegardé sous 'ablation_study_results.png'.")
    plt.show()