# two entangled input states
bosonic_state = pcvl.StateVector("|0,0,0,{A:1},{A:2},0,0,0>") + pcvl.StateVector("|0,0,0,{A:2},{A:1},0,0,0>")
fermionic_state = pcvl.StateVector("|0,0,0,{A:1},{A:2},0,0,0>") - pcvl.StateVector("|0,0,0,{A:2},{A:1},0,0,0>")


# select a backend and define the simulator on the circuit
simulator = Simulator(SLOSBackend())
simulator.set_circuit(circuit)

bosonic_prob_dist = simulator.probs(bosonic_state)
fermionic_prob_dist = simulator.probs(fermionic_state)

print("bosonic output distribution:", bosonic_prob_dist)
print("fermionic output distribution:", fermionic_prob_dist)

bosonic_modes = [get_mode(state) for state, _ in bosonic_prob_dist.items()]
bosonic_modes = [m if isinstance(m, list) else [m,m] for m in bosonic_modes]
fermionic_modes = [get_mode(state) for state, _ in fermionic_prob_dist.items()]
fermionic_modes = [m if isinstance(m, list) else [m,m] for m in fermionic_modes]

# get the probabilities of the modes
bosonic_probs = np.array([[0]*n]*n, dtype=np.float64)
for m, (_, prob) in zip(bosonic_modes, bosonic_prob_dist.items()):
    bosonic_probs[m[0], m[1]] = prob

fermionic_probs = np.array([[0]*n]*n, dtype=np.float64)
for m, (_, prob) in zip(fermionic_modes, fermionic_prob_dist.items()):
    fermionic_probs[m[0], m[1]] = prob

# get the walk positions distributions
walk_pos = range(-steps, steps+1)

bosonic_walk_probs = np.array([[0]*(2*steps+1)]*(2*steps+1), dtype=np.float64)
fermionic_walk_probs = np.array([[0]*(2*steps+1)]*(2*steps+1), dtype=np.float64)
for i in range(n):
    for j in range(n):
        w_i = mode_to_walk_pos_mapping[i]+steps
        w_j = mode_to_walk_pos_mapping[j]+steps
        bosonic_walk_probs[w_i, w_j] += bosonic_probs[i,j]
        fermionic_walk_probs[w_i, w_j] += fermionic_probs[i,j]

x, y = np.meshgrid(walk_pos, walk_pos)
cmap = plt.get_cmap('jet') # Get desired colormap
bosonic_max_height = np.max(bosonic_walk_probs.flatten())
bosonic_min_height = np.min(bosonic_walk_probs.flatten())
fermionic_max_height = np.max(fermionic_walk_probs.flatten())
fermionic_min_height = np.min(fermionic_walk_probs.flatten())
# scale each z to [0,1], and get their rgb values
bosonic_rgba = [cmap((k-bosonic_min_height)/bosonic_max_height) if k!=0 else (0,0,0,0) for k in bosonic_walk_probs.flatten()]
fermionic_rgba =  [cmap((k-fermionic_min_height)/fermionic_max_height) if k!=0 else (0,0,0,0) for k in fermionic_walk_probs.flatten()]
fig = plt.figure(figsize=(10, 16))
ax = plt.subplot(1, 2, 1, projection='3d')
ax.bar3d(x.flatten(), y.flatten(), np.zeros((2*steps+1)*(2*steps+1)), 1, 1, bosonic_walk_probs.flatten(), color=bosonic_rgba)
ax.set_xlabel("position")
ax.set_ylabel("position")
ax.set_zlabel("probability")
ax.set_box_aspect(aspect=None, zoom=0.8)
ax.set_title("bosonic")
ax = plt.subplot(1, 2, 2, projection='3d')
ax.bar3d(x.flatten(), y.flatten(), np.zeros((2*steps+1)*(2*steps+1)), 1, 1, fermionic_walk_probs.flatten(), color=fermionic_rgba)
ax.set_xlabel("position")
ax.set_ylabel("position")
ax.set_zlabel("probability")
ax.set_box_aspect(aspect=None, zoom=0.8)
ax.set_title("fermionic")
plt.show()