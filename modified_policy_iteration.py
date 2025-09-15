import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

    for it in range(max_iter):

        # partial policy evaluation
        for _ in range(policy_iter):
            V_new = np.zeros_like(V)
            for i in range(N+1):
                if policy[i] == 1:
                    V_new[i] = -i # negative cost for directly accepting
                else:
                    V_new[i] = C + alpha*np.dot(P,V)

            V[:] = V_new
            

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

        if not improvment:
            break

    return V, policy, value_improvement, policy_improvement


if __name__ == "__main__":

    problem = SellingAssetProblem(N=1000, alpha=0.9, C=10)
    V, policy, V_hist, pi_hist = modified_policy_iteration(problem)

    theory_i_star = calculated_threshold(problem)
    print("Theoretical threshold i* =", theory_i_star)

    # Where RL learned cutoff:
    rl_i_star = np.min([i for i, a in enumerate(policy) if a==1])
    print("RL threshold i* =", rl_i_star)

    # Animate
    ani = animate_mpi(V_hist, pi_hist)




