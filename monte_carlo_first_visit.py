import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from scipy.stats import norm
from collections import defaultdict


def create_transition_matrix(N, sigma, revert_strength=0.1, market_mean=None, markov=True):
    if not markov:
        return np.ones((N+1, N+1))/(N+1) # same as uniform distribution
    
    if market_mean is None:
        market_mean = N / 2
        
    states = np.arange(N + 1)
    # P_matrix[i, j] will be the probability of transitioning from state i to state j
    P_matrix = np.zeros((N + 1, N + 1))
    
    for i in range(N + 1):
        # Calculate the mean for the next offer distribution (mean-reversion)
        mean_offer = (1 - revert_strength) * i + revert_strength * market_mean
        
        # Calculate the probability for each possible next state j
        # using a normal distribution PDF, then normalize.
        # This is a discretized normal distribution.
        P_matrix[i, :] = norm.pdf(states, loc=mean_offer, scale=sigma)
        
        # Normalize the row so that probabilities sum to 1
        row_sum = P_matrix[i, :].sum()
        if row_sum > 0:
            P_matrix[i, :] /= row_sum
        else: # Handle edge case where all probabilities are zero
            P_matrix[i, i] = 1.0

    return P_matrix

class SellingAssetProblem:
    def __init__(self,N=100,alpha=0.5,C=10,sigma=15, P=None, markov=True):
        self.N = N
        self.alpha = alpha
        self.C = C
        self.P = P if P is not None else (create_transition_matrix(N,sigma, market_mean=N/2))
        self.states = list(range(N+1))
        self.actions = [0,1] # accept(1) and reject(0)
        self.policy = np.random.randint(0,2,size=N+1)
        self.state = None


    def take_action(self, action):
        if action == 0:
            cost = -self.state
            done = True
            next_state = None 
        else: 
            cost = self.C
            done = False
            next_state = np.random.choice(self.states,p=self.P)

        self.state = next_state
        return next_state, cost, done
    
    def get_actions(self):
        return self.actions
    
    def get_all_states(self):
        return self.states
    
    def is_end(self):
        return self.state is None
    

def generate_episodes(problem: SellingAssetProblem, policy: np.array, num_episodes: int):
    all_episodes = []
    C = problem.C
    for _ in range(num_episodes):
        current_episode = []
        current_state = np.random.choice(problem.states)
        
        while True:
            action = policy[current_state]
            if action == 1:
                cost = -current_state
                current_episode.append((current_state, action, cost))
                break
            else: 
                cost = C 
                current_episode.append((current_state, action, cost))
                transition_probabilites = problem.P[current_state]
                next_state = np.random.choice(problem.states, p = transition_probabilites)
                current_state = next_state
        all_episodes.append(current_episode)

    return all_episodes



def first_visit_monte_carlo(problem: SellingAssetProblem, policy: np.array, num_episodes=100, first_visit=True):
    num_states = problem.N 

    if(len(policy)-1 != num_states):
        raise ValueError("num_states and number of states in policy should match")
    
    returns_list = defaultdict(list)
    V = np.zeros(num_states+1)

    all_episodes = generate_episodes(problem, policy, num_episodes)
    
    for episode in all_episodes:
        if first_visit:
            states_visited_in_episode = set()
        G = 0

        for state, action , cost in reversed(episode):
            G = cost + problem.alpha * G 

            if (first_visit and state not in states_visited_in_episode) or (not first_visit):
                returns_list[state].append(G)
                V[state] = np.mean(returns_list[state])
                if first_visit:
                    states_visited_in_episode.add(state)

    
    return V, policy
    

def temporal_difference_learning(problem: SellingAssetProblem, policy: np.array, num_episodes=1000, n=0, alpha_lr = 0.1):
    num_states = problem.N
    gamma = problem.alpha 

    V = np.zeros(num_states + 1)

    all_episodes = generate_episodes(problem, policy, num_episodes)

    for episode in all_episodes:
        T = len(episode)
        for t in range(T):
            G = 0
            if n == 0 or n == float('inf'):
                if n == 0:
                    n_end = min(t+1, T)
                else: 
                    n_end = T

            else:
                n_end = min(t+n, T)


            for i in range(t, n_end):
                state_i, action_i, cost_i = episode[i]
                G += (gamma**(i-t))* cost_i

            if n_end < T:
                state_end = episode[n_end][0]
                G += (gamma ** (n_end - t)) * V[state_end]

            state_t = episode[t][0]
            V[state_t] = V[state_t] + alpha_lr * (G - V[state_t])

    return V, policy

def compare_td_methods(problem, fixed_policy, n_values, num_episodes=5000):
    """Compare TD(n) for different values of n"""
    
    print("Calculating true value function using Policy Evaluation...")
    V_true = policy_evaluation(problem, fixed_policy)
    
    results = {}
    
    print("\nRunning TD(n) for different values of n...")
    for n in n_values:
        n_runs = 10
        V_estimates = []
        
        for _ in range(n_runs):
            if n == float('inf'):
                # Use Monte Carlo for TD(∞)
                V_td, _ = first_visit_monte_carlo(problem, fixed_policy, num_episodes)
            else:
                V_td, _ = temporal_difference_learning(problem, fixed_policy, num_episodes, n=n, alpha_lr=0.1)
            V_estimates.append(V_td)
        
        V_td_mean = np.mean(V_estimates, axis=0)
        
        # Calculate error metrics
        mse = np.mean((V_true - V_td_mean)**2)
        mae = np.mean(np.abs(V_true - V_td_mean))
        max_err = np.max(np.abs(V_true - V_td_mean))
        
        results[n] = {
            'V_mean': V_td_mean,
            'mse': mse,
            'mae': mae,
            'max_error': max_err
        }
        
        n_str = '∞' if n == float('inf') else str(n)
        print(f"  TD({n_str:>3}): MSE={mse:8.4f}, MAE={mae:8.4f}, Max={max_err:8.4f}")
    
    return results, V_true


def plot_td_comparison(results, V_true, n_values):
    """Plot comparison of TD(n) methods"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Value functions
    ax = axes[0]
    ax.plot(V_true, label='True Value (DP)', color='black', linewidth=3)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_values)))
    for i, n in enumerate(n_values):
        n_str = '∞' if n == float('inf') else str(n)
        ax.plot(results[n]['V_mean'], label=f'TD({n_str})', 
                linestyle='--', alpha=0.7, color=colors[i])
    
    ax.set_xlabel('State (Offer)')
    ax.set_ylabel('Value')
    ax.set_title('Value Function: TD(n) Comparison')
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    
    # Plot 2: Error metrics
    ax = axes[1]
    n_labels = ['∞' if n == float('inf') else str(n) for n in n_values]
    mse_values = [results[n]['mse'] for n in n_values]
    mae_values = [results[n]['mae'] for n in n_values]
    
    x = np.arange(len(n_values))
    width = 0.35
    
    ax.bar(x - width/2, mse_values, width, label='MSE', alpha=0.8)
    ax.bar(x + width/2, mae_values, width, label='MAE', alpha=0.8)
    
    ax.set_xlabel('n (TD step size)')
    ax.set_ylabel('Error')
    ax.set_title('Error Comparison: TD(n)')
    ax.set_xticks(x)
    ax.set_xticklabels(n_labels)
    ax.legend()
    ax.grid(True, axis='y', ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('td_n_comparison.png', dpi=300)
    print("\nPlot saved as 'td_n_comparison.png'")
    plt.show()


def policy_evaluation(problem: SellingAssetProblem, policy=None, policy_iter=1,theta=1e-8):
    N = problem.N 
    policy = policy if policy is not None else problem.policy
    P = problem.P 
    alpha = problem.alpha
    C = problem.C


    V = np.zeros(N+1)

    while True: 
        delta = 0 

        for s in range(0,N+1):
            v_old = V[s]

            action = policy[s]
            if action == 1:
                v_new = -s 
            else: 
                P_row = P[s,:]
                v_new = C + alpha*np.dot(P_row , V)

            V[s] = v_new 
            delta = max(delta, abs(v_old - v_new))

        if(delta < theta):
            break 
    
    return V

def compare_methods(problem, fixed_policy, episode_counts, first_visit=True):
    """Compare MC and DP with multiple metrics"""
    
    # Ground truth
    print("Calculating true value function using Policy Evaluation...")
    V_true = policy_evaluation(problem, fixed_policy)
    
    results = {
        'episodes': [],
        'mse': [],
        'max_error': [],
        'mean_abs_error': [],
        'rel_error': [],
        'std_over_runs': []
    }
    
    print("\nRunning Monte Carlo with increasing episodes...")
    for n_eps in episode_counts:
        # Run multiple times to get confidence intervals
        n_runs = 10
        V_estimates = []
        
        for _ in range(n_runs):
            V_mc, _ = first_visit_monte_carlo(problem, fixed_policy, n_eps, first_visit=first_visit)
            V_estimates.append(V_mc)
        
        V_mc_mean = np.mean(V_estimates, axis=0)
        V_mc_std = np.std(V_estimates, axis=0)
        
        # Calculate various error metrics
        mse = np.mean((V_true - V_mc_mean)**2)
        max_err = np.max(np.abs(V_true - V_mc_mean))
        mae = np.mean(np.abs(V_true - V_mc_mean))
        
        # Relative error (avoid division by zero)
        mask = np.abs(V_true) > 1e-6
        rel_err = np.mean(np.abs((V_true[mask] - V_mc_mean[mask]) / V_true[mask]))
        
        results['episodes'].append(n_eps)
        results['mse'].append(mse)
        results['max_error'].append(max_err)
        results['mean_abs_error'].append(mae)
        results['rel_error'].append(rel_err)
        results['std_over_runs'].append(np.mean(V_mc_std))
        
        print(f"  {n_eps:7d} episodes -> MSE: {mse:8.4f}, MAE: {mae:8.4f}, "
              f"Max: {max_err:8.4f}, Rel: {rel_err:8.4%}, Std: {np.mean(V_mc_std):8.4f}")
    
    return results, V_true


def plot_comprehensive_results(results, V_true, mc_results, fig_name='comprehensive_convergence_analysis.png'):
    """Create comprehensive convergence plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Multiple error metrics
    ax = axes[0, 0]
    ax.plot(results['episodes'], results['mse'], 'o-', label='MSE')
    ax.plot(results['episodes'], results['mean_abs_error'], 's-', label='MAE')
    ax.plot(results['episodes'], results['max_error'], '^-', label='Max Error')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Episodes')
    ax.set_ylabel('Error')
    ax.set_title('Convergence: Multiple Error Metrics')
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # Plot 2: Relative error and std
    ax = axes[0, 1]
    ax.plot(results['episodes'], results['rel_error'], 'o-', color='red', label='Relative Error')
    ax.set_xscale('log')
    ax.set_xlabel('Number of Episodes')
    ax.set_ylabel('Relative Error')
    ax.set_title('Relative Error vs Episodes')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    
    ax2 = ax.twinx()
    ax2.plot(results['episodes'], results['std_over_runs'], 's-', color='blue', label='Std Dev')
    ax2.set_ylabel('Standard Deviation', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.legend(loc='upper right')
    
    # Plot 3: Value function comparison
    ax = axes[1, 0]
    ax.plot(V_true, label='True Value (DP)', color='black', linewidth=3)
    plot_eps = [results['episodes'][0], results['episodes'][len(results['episodes'])//2], 
                results['episodes'][-1]]
    for n_eps in plot_eps:
        if n_eps in mc_results:
            ax.plot(mc_results[n_eps], label=f'MC ({n_eps} eps)', linestyle='--', alpha=0.7)
    ax.set_xlabel('State (Offer)')
    ax.set_ylabel('Value')
    ax.set_title('Value Function Comparison')
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    
    # Plot 4: Error by state (for largest episode count)
    ax = axes[1, 1]
    final_mc = mc_results[results['episodes'][-1]]
    errors_by_state = np.abs(V_true - final_mc)
    ax.plot(errors_by_state, 'o-', markersize=3)
    ax.set_xlabel('State (Offer)')
    ax.set_ylabel('Absolute Error')
    ax.set_title(f'Error by State ({results["episodes"][-1]} episodes)')
    ax.grid(True, ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)
    print(f"\nPlot saved as '{fig_name}'")


def statistical_convergence_test(problem, policy, V_true, n_episodes=10000, n_runs=30, alpha=0.05, first_visit=True):
    """Perform statistical test for convergence"""
    from scipy import stats
    
    estimates = []
    for _ in range(n_runs):
        V_mc, _ = first_visit_monte_carlo(problem, policy, n_episodes, first_visit=first_visit)
        estimates.append(V_mc)
    
    estimates = np.array(estimates)
    means = np.mean(estimates, axis=0)
    stds = np.std(estimates, axis=0)
    
    # Confidence intervals
    ci = stats.t.interval(1-alpha, n_runs-1, loc=means, scale=stds/np.sqrt(n_runs))
    
    # Check if true value is within CI
    within_ci = (V_true >= ci[0]) & (V_true <= ci[1])
    coverage = np.mean(within_ci)
    
    print(f"\nStatistical Convergence Test (n={n_episodes}, runs={n_runs}):")
    print(f"  Coverage (% of states where true value in {(1-alpha)*100}% CI): {coverage*100:.1f}%")
    print(f"  Mean absolute error: {np.mean(np.abs(V_true - means)):.4f}")
    print(f"  Mean standard error: {np.mean(stds/np.sqrt(n_runs)):.4f}")
    
    return coverage >= 0.95  # Should be close to 1-alpha

def compare_td_vs_mc_convergence(problem, fixed_policy, episode_counts, num_runs=10):
    """Compare convergence speed of TD(0), First-Visit MC, and Every-Visit MC"""
    
    print("\n" + "=" * 60)
    print("CONVERGENCE COMPARISON: TD(0) vs MC Methods")
    print("=" * 60)
    
    # Ground truth
    print("Calculating true value function using Policy Evaluation...")
    V_true = policy_evaluation(problem, fixed_policy)
    
    results = {
        'TD(0)': {'episodes': [], 'mse': [], 'mae': [], 'max_error': [], 'std': []},
        'First-Visit MC': {'episodes': [], 'mse': [], 'mae': [], 'max_error': [], 'std': []},
        'Every-Visit MC': {'episodes': [], 'mse': [], 'mae': [], 'max_error': [], 'std': []}
    }
    
    print("\nRunning comparisons...")
    for n_eps in episode_counts:
        print(f"\n  Episodes: {n_eps}")
        
        # TD(0)
        V_td_estimates = []
        for _ in range(num_runs):
            V_td, _ = temporal_difference_learning(problem, fixed_policy, n_eps, n=0, alpha_lr=0.1)
            V_td_estimates.append(V_td)
        
        V_td_mean = np.mean(V_td_estimates, axis=0)
        V_td_std = np.std(V_td_estimates, axis=0)
        
        td_mse = np.mean((V_true - V_td_mean)**2)
        td_mae = np.mean(np.abs(V_true - V_td_mean))
        td_max = np.max(np.abs(V_true - V_td_mean))
        
        results['TD(0)']['episodes'].append(n_eps)
        results['TD(0)']['mse'].append(td_mse)
        results['TD(0)']['mae'].append(td_mae)
        results['TD(0)']['max_error'].append(td_max)
        results['TD(0)']['std'].append(np.mean(V_td_std))
        
        print(f"    TD(0):         MSE={td_mse:8.4f}, MAE={td_mae:8.4f}, Max={td_max:8.4f}")
        
        # First-Visit MC
        V_fv_estimates = []
        for _ in range(num_runs):
            V_fv, _ = first_visit_monte_carlo(problem, fixed_policy, n_eps, first_visit=True)
            V_fv_estimates.append(V_fv)
        
        V_fv_mean = np.mean(V_fv_estimates, axis=0)
        V_fv_std = np.std(V_fv_estimates, axis=0)
        
        fv_mse = np.mean((V_true - V_fv_mean)**2)
        fv_mae = np.mean(np.abs(V_true - V_fv_mean))
        fv_max = np.max(np.abs(V_true - V_fv_mean))
        
        results['First-Visit MC']['episodes'].append(n_eps)
        results['First-Visit MC']['mse'].append(fv_mse)
        results['First-Visit MC']['mae'].append(fv_mae)
        results['First-Visit MC']['max_error'].append(fv_max)
        results['First-Visit MC']['std'].append(np.mean(V_fv_std))
        
        print(f"    First-Visit:   MSE={fv_mse:8.4f}, MAE={fv_mae:8.4f}, Max={fv_max:8.4f}")
        
        # Every-Visit MC
        V_ev_estimates = []
        for _ in range(num_runs):
            V_ev, _ = first_visit_monte_carlo(problem, fixed_policy, n_eps, first_visit=False)
            V_ev_estimates.append(V_ev)
        
        V_ev_mean = np.mean(V_ev_estimates, axis=0)
        V_ev_std = np.std(V_ev_estimates, axis=0)
        
        ev_mse = np.mean((V_true - V_ev_mean)**2)
        ev_mae = np.mean(np.abs(V_true - V_ev_mean))
        ev_max = np.max(np.abs(V_true - V_ev_mean))
        
        results['Every-Visit MC']['episodes'].append(n_eps)
        results['Every-Visit MC']['mse'].append(ev_mse)
        results['Every-Visit MC']['mae'].append(ev_mae)
        results['Every-Visit MC']['max_error'].append(ev_max)
        results['Every-Visit MC']['std'].append(np.mean(V_ev_std))
        
        print(f"    Every-Visit:   MSE={ev_mse:8.4f}, MAE={ev_mae:8.4f}, Max={ev_max:8.4f}")
    
    return results, V_true


def plot_method_comparison(results, V_true):
    """Plot comprehensive comparison of TD(0) vs MC methods"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    methods = ['TD(0)', 'First-Visit MC', 'Every-Visit MC']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    # Plot 1: MSE comparison
    ax = axes[0, 0]
    for i, method in enumerate(methods):
        ax.plot(results[method]['episodes'], results[method]['mse'], 
                marker=markers[i], color=colors[i], label=method, 
                linewidth=2, markersize=8, alpha=0.8)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Episodes', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    ax.set_title('MSE Convergence Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    # Plot 2: MAE comparison
    ax = axes[0, 1]
    for i, method in enumerate(methods):
        ax.plot(results[method]['episodes'], results[method]['mae'], 
                marker=markers[i], color=colors[i], label=method, 
                linewidth=2, markersize=8, alpha=0.8)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Episodes', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax.set_title('MAE Convergence Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    # Plot 3: Max Error comparison
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        ax.plot(results[method]['episodes'], results[method]['max_error'], 
                marker=markers[i], color=colors[i], label=method, 
                linewidth=2, markersize=8, alpha=0.8)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Episodes', fontsize=12)
    ax.set_ylabel('Max Absolute Error', fontsize=12)
    ax.set_title('Maximum Error Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    # Plot 4: Standard deviation comparison (uncertainty)
    ax = axes[1, 1]
    for i, method in enumerate(methods):
        ax.plot(results[method]['episodes'], results[method]['std'], 
                marker=markers[i], color=colors[i], label=method, 
                linewidth=2, markersize=8, alpha=0.8)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Episodes', fontsize=12)
    ax.set_ylabel('Mean Standard Deviation', fontsize=12)
    ax.set_title('Uncertainty Comparison (Std Dev)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('td_vs_mc_convergence_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved as 'td_vs_mc_convergence_comparison.png'")
    plt.show()


def create_convergence_summary_table(results):
    """Create a summary table showing which method is best at each episode count"""
    
    print("\n" + "=" * 80)
    print("CONVERGENCE SUMMARY TABLE")
    print("=" * 80)
    
    methods = ['TD(0)', 'First-Visit MC', 'Every-Visit MC']
    episode_counts = results['TD(0)']['episodes']
    
    print(f"\n{'Episodes':<12} {'Best MSE':<20} {'Best MAE':<20} {'Best Max Error':<20}")
    print("-" * 80)
    
    for i, n_eps in enumerate(episode_counts):
        # Find best method for each metric
        mse_values = {method: results[method]['mse'][i] for method in methods}
        mae_values = {method: results[method]['mae'][i] for method in methods}
        max_values = {method: results[method]['max_error'][i] for method in methods}
        
        best_mse = min(mse_values, key=mse_values.get)
        best_mae = min(mae_values, key=mae_values.get)
        best_max = min(max_values, key=max_values.get)
        
        print(f"{n_eps:<12} {best_mse:<20} {best_mae:<20} {best_max:<20}")
    
    print("=" * 80)
    
    # Overall winner (method that appears most often)
    from collections import Counter
    winners = []
    for i in range(len(episode_counts)):
        mse_values = {method: results[method]['mse'][i] for method in methods}
        mae_values = {method: results[method]['mae'][i] for method in methods}
        max_values = {method: results[method]['max_error'][i] for method in methods}
        
        winners.append(min(mse_values, key=mse_values.get))
        winners.append(min(mae_values, key=mae_values.get))
        winners.append(min(max_values, key=max_values.get))
    
    winner_counts = Counter(winners)
    overall_winner = winner_counts.most_common(1)[0][0]
    
    print(f"\n🏆 OVERALL WINNER: {overall_winner}")
    print(f"   (Best performance in {winner_counts[overall_winner]} out of {len(winners)} comparisons)")
    print("=" * 80)


if __name__ == "__main__":
    N_STATES = 100
    THRESHOLD = 60
    fixed_policy = np.array([0 if i < THRESHOLD else 1 for i in range(N_STATES + 1)])
    
    problem = SellingAssetProblem(N=N_STATES, C=10, sigma=20, alpha=0.9)
    
    episode_counts = [100, 500, 2000, 10000, 50000]

    # print(f"----FIRST-VISIT MONTE CARLO COMPARISON----")    
    # results, V_true = compare_methods(problem, fixed_policy, episode_counts)
    
    # # Store MC results for plotting
    # mc_results = {}
    # for n_eps in episode_counts:
    #     V_mc, _ = first_visit_monte_carlo(problem, fixed_policy, n_eps)
    #     mc_results[n_eps] = V_mc
    
    # plot_comprehensive_results(results, V_true, mc_results, fig_name='first_visit_comprehensive_convergence_analysis.png')
    
    # # Statistical test
    # converged = statistical_convergence_test(problem, fixed_policy, V_true)
    # print(f"\nConvergence achieved: {converged}")

    # # Every-visit MC comparison
    # print(f"EVERY-VISIT MONTE CARLO COMPARISON")

    # results, V_true = compare_methods(problem, fixed_policy, episode_counts, first_visit=False)
    
    # # Store MC results for plotting
    # mc_results = {}
    # for n_eps in episode_counts:
    #     V_mc, _ = first_visit_monte_carlo(problem, fixed_policy, n_eps, first_visit=False)
    #     mc_results[n_eps] = V_mc

    # plot_comprehensive_results(results, V_true, mc_results, fig_name='every_visit_comprehensive_convergence_analysis.png')

    # # Statistical test
    # converged = statistical_convergence_test(problem, fixed_policy, V_true, first_visit=False)
    # print(f"\nConvergence achieved: {converged}")


    # TD(n) comparison
    print("\n" + "=" * 60)
    print("TD(n) COMPARISON")
    print("=" * 60)
    n_values = [0, 1, 3, 5, 10, 20, float('inf')]  # Different n-step values
    td_results, V_true = compare_td_methods(problem, fixed_policy, n_values, num_episodes=5000)
    plot_td_comparison(td_results, V_true, n_values)

    comparison_results, V_true = compare_td_vs_mc_convergence(problem, fixed_policy, episode_counts, num_runs=10)
    plot_method_comparison(comparison_results, V_true)
    create_convergence_summary_table(comparison_results)
