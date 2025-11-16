# Grid World - Solution avec Value Iteration et Policy Iteration

##  Description du problème

Ce projet résout le problème **Grid World** classique en apprentissage par renforcement en utilisant deux algorithmes de programmation dynamique:
1. **Value Iteration**
2. **Policy Iteration**

### Environnement
- **Grille**: 4x4 (16 états numérotés de 0 à 15)
- **États terminaux**: 0 et 15 (cases grises)
- **Actions possibles**: UP (0), RIGHT (1), DOWN (2), LEFT (3)
- **Récompense**: -1 à chaque action (sauf états terminaux)
- **Objectif**: Trouver la politique optimale pour atteindre un état terminal

### Paramètres testés
- **γ (gamma)**: Facteur de discount = 1.0 et 0.9
- **θ (theta)**: Critère de convergence = 10^-4

## Installation

### Prérequis
```bash
pip install numpy matplotlib seaborn jupyter
```

### Bibliothèques nécessaires
- `numpy`: Calculs matriciels
- `matplotlib`: Visualisation des graphiques
- `seaborn`: Heatmaps et visualisations avancées
- `jupyter`: Environnement de notebook

##  Fichiers fournis

1. **grid_world_solution.ipynb**: Notebook principal avec l'implémentation complète
2. **README.md**: Ce fichier d'instructions

##  Utilisation

### Lancer le notebook
```bash
jupyter notebook grid_world_solution.ipynb
```

### Structure du notebook

Le notebook est organisé en 10 sections:

1. **Setup et imports**: Installation des bibliothèques
2. **Environnement Grid World**: Classe définissant la grille et les transitions
3. **Value Iteration**: Implémentation de l'algorithme
4. **Policy Iteration**: Implémentation avec évaluation et amélioration
5. **Visualisation**: Fonctions pour afficher les résultats
6. **Tests avec γ = 1.0**: Exécution des deux algorithmes
7. **Tests avec γ = 0.9**: Exécution des deux algorithmes
8. **Comparaison**: Analyse comparative des résultats
9. **Convergence**: Visualisation de la convergence
10. **Sauvegarde**: Export des résultats

### Exécution rapide

Pour exécuter tout le notebook d'un coup:
- **Jupyter**: Menu → Cell → Run All
- **Ou**: Shift + Enter pour chaque cellule

## Résultats attendus

### Value Iteration
- Converge en ~3-4 itérations avec γ = 1.0
- Converge en ~4-5 itérations avec γ = 0.9
- Trouve la fonction de valeur optimale V*
- Extrait la politique optimale π*

### Policy Iteration
- Converge en ~2-3 itérations
- Plus rapide que Value Iteration en nombre d'itérations
- Trouve la même politique optimale

### Visualisations générées
1. **Heatmaps**: Fonction de valeur pour chaque état
2. **Grilles avec flèches**: Politique optimale (direction à prendre)
3. **Graphiques de convergence**: Évolution des valeurs au fil des itérations

### Politique optimale typique

Pour γ = 1.0, la politique guide l'agent vers l'état terminal le plus proche:
```
T  ←  ←  ↓
↑  ↑  ←  ↓
↑  →  →  ↓
↑  →  →  T
```

Où:
- T = Terminal (états 0 et 15)
- Flèches = Direction optimale

##  Interprétation des résultats

### Fonction de valeur
- **Valeurs négatives**: Reflètent le nombre d'étapes (avec récompense -1) jusqu'à l'état terminal
- **États terminaux**: Valeur = 0
- **États proches des terminaux**: Valeurs moins négatives

### Impact de γ
- **γ = 1.0**: L'agent valorise également les récompenses futures et présentes
- **γ = 0.9**: L'agent préfère les récompenses immédiates (plus myope)

### Comparaison des algorithmes
- Les deux convergent vers la même solution
- Policy Iteration converge plus rapidement en itérations
- Value Iteration peut être plus simple à implémenter

##  Personnalisation

### Modifier les paramètres

Dans le notebook, vous pouvez facilement changer:

```python
# Changer le facteur de discount
gamma = 0.95  # Valeur entre 0 et 1

# Changer le critère de convergence
theta = 1e-6  # Plus petit = plus précis mais plus lent

# Tester sur une grille plus grande
env = GridWorld(size=5)  # Grille 5x5
```

### Ajouter des obstacles

Modifier la classe `GridWorld` pour bloquer certaines cases:

```python
class GridWorld:
    def __init__(self, size=4):
        # ... code existant ...
        self.obstacles = [5, 6]  # États bloqués
        
    def get_next_state(self, state, action):
        next_state = # ... calcul ...
        if next_state in self.obstacles:
            return state  # Reste sur place
        return next_state
```

## Questions du projet

Le notebook répond aux deux questions posées:

### Question 1: Value Iteration
Implémenté dans la section 2
- Retourne la politique optimale
- Retourne la fonction de valeur correspondante

### Question 2: Policy Iteration
 Implémenté dans la section 3
- Policy Evaluation: Évalue une politique donnée
- Policy Improvement: Améliore la politique
- Retourne la politique optimale et la fonction de valeur

##  Dépannage

### Erreur d'import
```bash
ModuleNotFoundError: No module named 'seaborn'
```
**Solution**: `pip install seaborn`

### Graphiques ne s'affichent pas
**Solution**: Ajouter `%matplotlib inline` au début du notebook

### Convergence trop lente
**Solution**: Augmenter theta ou vérifier que γ < 1

## Références

- **Sutton & Barto** - "Reinforcement Learning: An Introduction"
- **Bellman Equation** - Équation fondamentale du RL
- **Dynamic Programming** - Méthodes de résolution exacte

##  Auteur

Solution développée pour le cours d'apprentissage par renforcement.

## Licence

Code libre d'utilisation à des fins éducatives.
