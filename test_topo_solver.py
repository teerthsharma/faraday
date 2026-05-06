import numpy as np
from faraday.god_tensor import GodTensor
from faraday.topological_solver import TopologicalFDFD, TopologicalFDTD

# Train a dummy GodTensor
gt = GodTensor(n_geometries=10)
gt.collect_training_data(nx=20, ny=20, num_modes=2)
gt.learn_T()
gt.find_fixed_point()

# Test FDFD
fdfd = TopologicalFDFD(gt)
res = fdfd.solve((2.0, 1.5))
print("FDFD Solve:")
print(f"  Coupling: {res['predicted_coupling_score']:.4f}")
print(f"  E_Latent shape: {np.array(res['e_latent_vector']).shape}")

# Test FDTD
fdtd = TopologicalFDTD(gt, dt=0.01)
initial_state = np.array(res['e_latent_vector'])
history = fdtd.simulate(initial_state, steps=5)
print(f"FDTD Simulate (5 steps):")
for i, state in enumerate(history):
    print(f"  Step {i} norm: {np.linalg.norm(state):.4f}")
