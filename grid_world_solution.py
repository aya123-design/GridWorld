

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# Configuration de l'affichage
plt.style.use('seaborn-v0_8-darkgrid')


class GridWorld:
    """
    Environnement Grid World 4x4
    - États: 0 à 15
    - États terminaux: 0 et 15 (cases grises)
    - Actions: UP=0, RIGHT=1, DOWN=2, LEFT=3
    - Récompense: -1 pour chaque action (sauf états terminaux)
    """
    
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        
        # États terminaux
        self.terminal_states = [0, 15]
        
        # Actions: UP=0, RIGHT=1, DOWN=2, LEFT=3
        self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        self.action_effects = {
            0: (-1, 0),   # UP
            1: (0, 1),    # RIGHT
            2: (1, 0),    # DOWN
            3: (0, -1)    # LEFT
        }
        
    def state_to_position(self, state):
        """Convertit un numéro d'état en position (row, col)"""
        return state // self.size, state % self.size
    
    def position_to_state(self, row, col):
        """Convertit une position (row, col) en numéro d'état"""
        return row * self.size + col
    
    def is_terminal(self, state):
        """Vérifie si un état est terminal"""
        return state in self.terminal_states
    
    def get_next_state(self, state, action):
        """
        Retourne le prochain état après avoir pris une action.
        Si l'action mène hors de la grille, l'agent reste sur place.
        """
        # Si état terminal, reste sur place
        if self.is_terminal(state):
            return state
        
        row, col = self.state_to_position(state)
        d_row, d_col = self.action_effects[action]
        
        new_row = row + d_row
        new_col = col + d_col
        
        # Vérifier les limites de la grille
        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            return self.position_to_state(new_row, new_col)
        else:
            # Hors limites, reste sur place
            return state
    
    def get_reward(self, state, action, next_state):
        """
        Retourne la récompense pour une transition.
        Récompense de -1 pour chaque action sauf si on est déjà dans un état terminal.
        """
        if self.is_terminal(state):
            return 0
        return -1


def value_iteration(env, gamma=1.0, theta=1e-4, max_iterations=1000):
    """
    Algorithme Value Iteration
    
    Paramètres:
    - env: environnement GridWorld
    - gamma: facteur de discount
    - theta: critère de convergence
    - max_iterations: nombre maximum d'itérations
    
    Retourne:
    - V: fonction de valeur optimale
    - policy: politique optimale
    - history: historique des valeurs pour visualisation
    """
    
    # Initialiser la fonction de valeur à zéro
    V = np.zeros(env.n_states)
    history = [V.copy()]
    
    print("\n=== VALUE ITERATION ===")
    print(f"Gamma: {gamma}, Theta: {theta}\n")
    
    for iteration in range(max_iterations):
        delta = 0
        V_old = V.copy()
        
        # Pour chaque état
        for s in range(env.n_states):
            # Si état terminal, valeur = 0
            if env.is_terminal(s):
                V[s] = 0
                continue
            
            # Calculer la valeur pour chaque action possible
            action_values = []
            for a in range(env.n_actions):
                next_state = env.get_next_state(s, a)
                reward = env.get_reward(s, a, next_state)
                value = reward + gamma * V_old[next_state]
                action_values.append(value)
            
            # Prendre le maximum (Bellman optimality equation)
            V[s] = max(action_values)
            
            # Mettre à jour delta
            delta = max(delta, abs(V[s] - V_old[s]))
        
        history.append(V.copy())
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration:3d} | Delta: {delta:.6f}")
        
        # Critère de convergence
        if delta < theta:
            print(f"\n✓ Convergence atteinte après {iteration + 1} itérations")
            print(f"  Delta final: {delta:.8f}")
            break
    
    # Extraire la politique optimale
    policy = np.zeros(env.n_states, dtype=int)
    
    for s in range(env.n_states):
        if env.is_terminal(s):
            policy[s] = -1  # Pas d'action pour états terminaux
            continue
        
        action_values = []
        for a in range(env.n_actions):
            next_state = env.get_next_state(s, a)
            reward = env.get_reward(s, a, next_state)
            value = reward + gamma * V[next_state]
            action_values.append(value)
        
        # Politique greedy
        policy[s] = np.argmax(action_values)
    
    return V, policy, history


def policy_evaluation(env, policy, gamma=1.0, theta=1e-4, max_iterations=1000):
    """
    Évaluation de politique: calcule V^π pour une politique donnée
    """
    V = np.zeros(env.n_states)
    
    for iteration in range(max_iterations):
        delta = 0
        V_old = V.copy()
        
        for s in range(env.n_states):
            if env.is_terminal(s):
                V[s] = 0
                continue
            
            # Suivre la politique actuelle
            a = policy[s]
            next_state = env.get_next_state(s, a)
            reward = env.get_reward(s, a, next_state)
            V[s] = reward + gamma * V_old[next_state]
            
            delta = max(delta, abs(V[s] - V_old[s]))
        
        if delta < theta:
            break
    
    return V


def policy_improvement(env, V, gamma=1.0):
    """
    Amélioration de politique: trouve une politique greedy par rapport à V
    """
    policy = np.zeros(env.n_states, dtype=int)
    policy_stable = True
    
    for s in range(env.n_states):
        if env.is_terminal(s):
            policy[s] = -1
            continue
        
        old_action = policy[s]
        
        # Trouver la meilleure action
        action_values = []
        for a in range(env.n_actions):
            next_state = env.get_next_state(s, a)
            reward = env.get_reward(s, a, next_state)
            value = reward + gamma * V[next_state]
            action_values.append(value)
        
        policy[s] = np.argmax(action_values)
        
        if old_action != policy[s]:
            policy_stable = False
    
    return policy, policy_stable


def policy_iteration(env, gamma=1.0, theta=1e-4, max_iterations=100):
    """
    Algorithme Policy Iteration
    
    Retourne:
    - V: fonction de valeur optimale
    - policy: politique optimale
    - history: historique des politiques
    """
    
    # Initialiser avec une politique aléatoire
    policy = np.random.randint(0, env.n_actions, size=env.n_states)
    for s in env.terminal_states:
        policy[s] = -1
    
    history = []
    
    print("\n=== POLICY ITERATION ===")
    print(f"Gamma: {gamma}, Theta: {theta}\n")
    
    for iteration in range(max_iterations):
        # 1. Policy Evaluation
        V = policy_evaluation(env, policy, gamma, theta)
        
        # 2. Policy Improvement
        policy, policy_stable = policy_improvement(env, V, gamma)
        
        history.append((V.copy(), policy.copy()))
        
        print(f"Iteration {iteration + 1:2d} | Policy stable: {policy_stable}")
        
        # Si la politique est stable, on a convergé
        if policy_stable:
            print(f"\n✓ Convergence atteinte après {iteration + 1} itérations")
            break
    
    return V, policy, history


def plot_value_function(V, env, title="Value Function"):
    """
    Affiche la fonction de valeur sous forme de heatmap
    """
    V_grid = V.reshape(env.size, env.size)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(V_grid, annot=True, fmt=".2f", cmap="RdYlGn", 
                cbar_kws={'label': 'Value'}, square=True,
                linewidths=2, linecolor='black')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Column', fontsize=12)
    plt.ylabel('Row', fontsize=12)
    
    # Marquer les états terminaux
    for s in env.terminal_states:
        row, col = env.state_to_position(s)
        plt.gca().add_patch(Rectangle((col, row), 1, 1, 
                                      fill=False, edgecolor='blue', 
                                      linewidth=4))
    
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_policy(policy, env, title="Optimal Policy"):
    """
    Affiche la politique avec des flèches
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Dessiner la grille
    for i in range(env.size + 1):
        ax.plot([0, env.size], [i, i], 'k-', linewidth=2)
        ax.plot([i, i], [0, env.size], 'k-', linewidth=2)
    
    action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←', -1: '✓'}
    
    # Dessiner les actions
    for s in range(env.n_states):
        row, col = env.state_to_position(s)
        
        # Coordonnées du centre de la cellule
        x = col + 0.5
        y = env.size - row - 0.5
        
        # Marquer les états terminaux
        if env.is_terminal(s):
            ax.add_patch(Rectangle((col, env.size - row - 1), 1, 1,
                                  facecolor='lightgray', edgecolor='blue',
                                  linewidth=3))
            ax.text(x, y, f'T\n{s}', ha='center', va='center',
                   fontsize=14, fontweight='bold', color='blue')
        else:
            # Afficher le numéro d'état
            ax.text(x, y + 0.25, str(s), ha='center', va='center',
                   fontsize=10, color='gray')
            
            # Afficher la flèche d'action
            action = policy[s]
            ax.text(x, y - 0.05, action_symbols[action], 
                   ha='center', va='center',
                   fontsize=24, fontweight='bold', color='darkblue')
    
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_results(V, policy, env, algorithm_name):
    """
    Affiche les résultats sous forme de texte
    """
    print(f"\n{'='*60}")
    print(f"RÉSULTATS - {algorithm_name}")
    print(f"{'='*60}\n")
    
    print("Fonction de valeur optimale:")
    print("-" * 40)
    V_grid = V.reshape(env.size, env.size)
    for row in range(env.size):
        for col in range(env.size):
            state = env.position_to_state(row, col)
            print(f"{V_grid[row, col]:7.2f}", end=" ")
        print()
    
    print("\nPolitique optimale:")
    print("-" * 40)
    action_symbols = {0: '↑ ', 1: '→ ', 2: '↓ ', 3: '← ', -1: 'T '}
    policy_grid = policy.reshape(env.size, env.size)
    for row in range(env.size):
        for col in range(env.size):
            print(f"  {action_symbols[policy_grid[row, col]]}  ", end=" ")
        print()
    print()


def main():
    """
    Fonction principale pour exécuter les deux algorithmes
    """
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + " "*10 + "GRID WORLD - SOLUTION COMPLÈTE" + " "*17 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    # Créer l'environnement
    env = GridWorld(size=4)
    print(f"\nEnvironnement créé: {env.n_states} états, {env.n_actions} actions")
    print(f"États terminaux: {env.terminal_states}")
    
    # Test avec gamma = 1.0
    print("\n\n" + "="*70)
    print("TEST AVEC γ = 1.0")
    print("="*70)
    
    gamma = 1.0
    theta = 1e-4
    
    # Value Iteration
    V_vi_1, policy_vi_1, _ = value_iteration(env, gamma=gamma, theta=theta)
    print_results(V_vi_1, policy_vi_1, env, "VALUE ITERATION (γ=1.0)")
    plot_value_function(V_vi_1, env, "Value_Function_VI_gamma1.0")
    plot_policy(policy_vi_1, env, "Policy_VI_gamma1.0")
    
    # Policy Iteration
    V_pi_1, policy_pi_1, _ = policy_iteration(env, gamma=gamma, theta=theta)
    print_results(V_pi_1, policy_pi_1, env, "POLICY ITERATION (γ=1.0)")
    plot_value_function(V_pi_1, env, "Value_Function_PI_gamma1.0")
    plot_policy(policy_pi_1, env, "Policy_PI_gamma1.0")
    
    # Test avec gamma = 0.9
    print("\n\n" + "="*70)
    print("TEST AVEC γ = 0.9")
    print("="*70)
    
    gamma = 0.9
    
    # Value Iteration
    V_vi_09, policy_vi_09, _ = value_iteration(env, gamma=gamma, theta=theta)
    print_results(V_vi_09, policy_vi_09, env, "VALUE ITERATION (γ=0.9)")
    plot_value_function(V_vi_09, env, "Value_Function_VI_gamma0.9")
    plot_policy(policy_vi_09, env, "Policy_VI_gamma0.9")
    
    # Policy Iteration
    V_pi_09, policy_pi_09, _ = policy_iteration(env, gamma=gamma, theta=theta)
    print_results(V_pi_09, policy_pi_09, env, "POLICY ITERATION (γ=0.9)")
    plot_value_function(V_pi_09, env, "Value_Function_PI_gamma0.9")
    plot_policy(policy_pi_09, env, "Policy_PI_gamma0.9")
    
    # Comparaison
    print("\n" + "="*70)
    print("COMPARAISON DES ALGORITHMES")
    print("="*70)
    
    print("\nAvec γ = 1.0:")
    print("-" * 70)
    print(f"Différence max entre les fonctions de valeur: {np.max(np.abs(V_vi_1 - V_pi_1)):.10f}")
    print(f"Politiques identiques: {np.array_equal(policy_vi_1, policy_pi_1)}")
    
    print("\nAvec γ = 0.9:")
    print("-" * 70)
    print(f"Différence max entre les fonctions de valeur: {np.max(np.abs(V_vi_09 - V_pi_09)):.10f}")
    print(f"Politiques identiques: {np.array_equal(policy_vi_09, policy_pi_09)}")
    
    print("\n✓ Exécution terminée avec succès!")
    print("✓ Graphiques sauvegardés en PNG dans le répertoire courant")


if __name__ == "__main__":
    main()
