# Time Series Forecasting : LSTM "From Scratch"

Ce projet implémente une architecture de Deep Learning (LSTM) entièrement "from scratch" en utilisant uniquement la librairie Python classique `numpy`. 

## Objectifs atteints

- **Modularité :** Séparation entre les fonctions mathématiques, la cellule LSTM (locale) et le modèle temporel (BackPropagation Through Time "BPTT" global).
- **Initialisation justifiée :** Utilisation de l'initialisation Xavier/Glorot (adaptée aux fonctions Sigmoïde et Tanh) et initialisation du biais de la Forget Gate à 1 pour contrer le *Vanishing Gradient*.
- **Ablation Study :** Comparaison de la dynamique de convergence avec et sans la *Forget Gate* sur une série temporelle cyclique.

## Comment reproduire l'expérience (Évaluation Live)

1. **Prérequis :** Assurez-vous d'avoir Python installé ainsi que les bibliothèques de base.
   ```bash
   pip install numpy matplotlib
   ```

2. **Lancement de l'entraînement et de l'Ablation Study :**
   ```bash
   python main.py
   ```
   *Note : Le script générera automatiquement les données (une sinusoïde), entraînera les deux versions du modèle (avec et sans Forget Gate), affichera les logs de la Loss (A - Y), et générera le graphique de comparaison.*

## Architecture du projet

*   `functions.py` : Fonctions d'activation et dérivées analytiques (Règle de la chaîne).
*   `lstm.py` : La cellule LSTM avec gestion rigoureuse des dimensions, produit de Hadamard et rétropropagation locale.
*   `model.py` : Le modèle temporel (Unrolling), accumulation des gradients temporels et mise à jour SGD (W = W - alpha * dW).
*   `main.py` : Génération du dataset cyclique, boucle d'entraînement et génération du graphe comparatif.

