import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from scipy.stats import norm

def create_transition_matrix(N, sigma, revert_strength=0.1, market_mean=None):
    """
    Creates a Markov transition matrix P[i, j] = P(s_next=j | s_curr=i).

    Args:
        N (int): The maximum offer.
        sigma (float): The volatility of offers.
        revert_strength (float): How strongly offers revert to the mean (0=no reversion).
        market_mean (int): The global market mean offer. If None, defaults to N/2.
    """
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

def plot_convergence(convergence_norms):
    plt.figure(figsize=(6,4))
    plt.plot(convergence_norms, marker='o')
    plt.yscale("log")  # since convergence is exponential
    plt.xlabel("Iteration")
    plt.ylabel(r"$\|V_{n+1} - V_n\|_\infty$")
    plt.title("Convergence of Value Function")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()


def animate_mpi(value_improvement, policy_improvement, interval=1000):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    states = np.arange(len(value_improvement[0]))
    line, = ax1.plot([], [], lw=2)
    ax1.set_xlim(0, len(states)-1)
    ax1.set_ylim(np.min(value_improvement), np.max(value_improvement))
    ax1.set_title("Value Function Improvement")
    ax1.set_xlabel("State (Offer)")
    ax1.set_ylabel("Value (Cost)")

    (policy_line,) = ax2.step(states, policy_improvement[0], where="post", lw=2, color="blue")
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_title("Policy Improvement (0=Reject, 1=Accept)")
    ax2.set_xlabel("State (Offer)")
    ax2.set_ylabel("Action")

    def init():
        line.set_data([], [])
        policy_line.set_data([], [])
        return line, policy_line

    def update(frame):
        V = value_improvement[frame]
        pi = policy_improvement[frame]

        line.set_data(states, V)
        policy_line.set_data(states, pi)

        ax1.set_title(f"Value Function Iteration {frame}")
        ax2.set_title(f"Policy Iteration {frame}")
        return line, policy_line

    ani = animation.FuncAnimation(fig, update, frames=len(value_improvement),
                                  init_func=init, blit=False, interval=interval, repeat=False)
    plt.tight_layout()
    plt.show()
    return ani



class SellingAssetProblem:
    def __init__(self,N=100,alpha=0.5,C=10, P=None):
        self.N = N
        self.alpha = alpha
        self.C = C
        self.P = P if P is not None else np.ones(N + 1) / (N + 1)
        self.states = list(range(N+1))
        self.actions = [0,1] # accept(1) and reject(0)
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
    
def calculated_threshold(problem: SellingAssetProblem):
    N = max(problem.get_all_states())
    C = problem.C
    P = problem.P
    alpha = problem.alpha
    least_cost = float('inf')
    i_star = 1
    
    for i in range(1,N+1):
        sum_over_P = sum(P[j] for j in range(i))
        sum_over_jP = sum(j*P[j] for j in range(i,N+1))

        value = (C*sum_over_P - sum_over_jP)/(1- alpha*sum_over_P)
        if(value > 0): # positive cost, no point
            value = float('inf')

        if value < least_cost:
            least_cost = value
            i_star = i

    return i_star
        

def modified_policy_iteration(problem: SellingAssetProblem, max_iter= 1000, policy_iter=9,tol=1e-4):    
    print("policy iter" , policy_iter)
    N = max(problem.get_all_states())
    P = problem.P 
    alpha = problem.alpha
    C = problem.C

    # initialise vector V_o and pi_o (greedy wrt to V)
    V = np.zeros(N+1)
    policy = np.zeros(N+1)

    value_improvement = []
    policy_improvement = []
    convergence_norms = []

    start_time = time.time()

    for it in range(max_iter):

        V_old = V.copy()

        # partial policy evaluation
        for _ in range(policy_iter):
            V_new = np.zeros_like(V)
            for i in range(N+1):
                if policy[i] == 1:
                    V_new[i] = -i # negative cost for directly accepting
                else:
                    V_new[i] = C + alpha*np.dot(P,V)

            V[:] = V_new

        diff_norm = np.linalg.norm(V - V_old, ord=np.inf)  # sup norm
        convergence_norms.append(diff_norm)
            

        # greedy approach for policy improvement
        improvment = False
        for i in range(N+1):
            accept_cost = -i
            reject_cost = C + alpha*np.dot(P,V)
            new_action = 1 if accept_cost < reject_cost else 0
            if new_action != policy[i]:
                improvment = True
            policy[i] = new_action

        

        policy_improvement.append(policy.copy())
        value_improvement.append(V.copy())

        print(f"Iteration {it + 1}: Threshold = {next((s for s in range(N+1) if policy[s] == 1), 'None')}")

        if diff_norm < tol or not improvment:
            break

    end_time = time.time()
    runtime = end_time - start_time
    print(f"\nConverged in {it+1} iterations, time taken = {runtime:.4f} seconds")

    return V, policy, value_improvement, policy_improvement, convergence_norms, runtime

def mpi_finite_horizon(problem: SellingAssetProblem, T=100):
    N = max(problem.get_all_states())
    P = problem.P
    alpha = problem.alpha
    C = problem.C 
    V_table = np.zeros((T+1,N+1))
    policy_table = np.zeros((T, N + 1), dtype=int)

    # accept the offer on the last day (no other option)
    V_table[T, :] = -np.arange(N + 1)
    print("Starting backward induction...")
    start_time = time.time()

    for t in reversed(range(T)):
        reject_cost = C + alpha*np.dot(P,V_table[t+1])

        accept_costs = -np.arange(N+1)
        # The value at time t is the minimum of the two costs
        V_table[t, :] = np.minimum(accept_costs, reject_cost)

        # The policy is to accept (1) if the accept_cost is lower
        policy_table[t, :] = (accept_costs < reject_cost).astype(int)
        
        # Find and print the threshold for the current time step t
        threshold = next((s for s in range(N + 1) if policy_table[t, s] == 1), N)
        print(f"Time t={t}: Optimal threshold i*_{t} = {threshold}")

    end_time = time.time()
    runtime = end_time - start_time
    print(f"\nFinite horizon solution found in {runtime:.4f} seconds")
        
    return V_table, policy_table, runtime    

if __name__ == "__main__":

    problem = SellingAssetProblem(N=100, alpha=0.9, C=10)
    V, policy, V_hist, pi_hist, convergence_norms, runtime = modified_policy_iteration(problem)

    theory_i_star = calculated_threshold(problem)
    print("Theoretical threshold i* =", theory_i_star)

    # Where RL learned cutoff:
    rl_i_star = np.min([i for i, a in enumerate(policy) if a==1])
    print("RL threshold i* =", rl_i_star)

    # Animate
    ani = animate_mpi(V_hist, pi_hist)

    # Plot convergence
    plot_convergence(convergence_norms)

    # --- Finite Horizon ---
    print("--- SOLVING FOR FINITE HORIZON (T=50) ---")
    problem_fin = SellingAssetProblem(N=100, alpha=0.9, C=10)
    T = 6 # Example time horizon
    V_table_fin, policy_table_fin, runtime_fin = mpi_finite_horizon(problem_fin, T)
    
    # Visualize the time-dependent thresholds
    thresholds = [next((s for s, a in enumerate(pi) if a == 1), problem_fin.N) for pi in policy_table_fin]
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(T), thresholds, marker='o', linestyle='--')
    plt.title("Optimal Threshold vs. Time")
    plt.xlabel("Time Step (t)")
    plt.ylabel("Threshold (i*_t)")
    plt.grid(True)
    plt.show()





