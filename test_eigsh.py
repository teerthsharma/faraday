from faraday.em_solver import CavityGeometry, CavityShape, solve_cavity_modes

geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
res = solve_cavity_modes(geom, nx=20, ny=20, num_modes=3)
print(res["k_values"])
