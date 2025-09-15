import numpy as np
from modified_policy_iteration import modified_policy_iteration, SellingAssetProblem, animate_mpi,calculated_threshold


problem = SellingAssetProblem(N=1000, alpha=0.9, C=10)
V, policy, V_hist, pi_hist = modified_policy_iteration(problem,policy_iter=1)

theory_i_star = calculated_threshold(problem)
print("Theoretical threshold i* =", theory_i_star)

# Where RL learned cutoff:
rl_i_star = np.min([i for i, a in enumerate(policy) if a==1])
print("RL threshold i* =", rl_i_star)

# Animate
ani = animate_mpi(V_hist, pi_hist)