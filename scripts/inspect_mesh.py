import trimesh

# Load your STL file
mesh = trimesh.load("assets/objects/SodaCan.stl", force="mesh")

print("faces:", len(mesh.faces))
print("extents:", mesh.extents)
print("is_watertight:", mesh.is_watertight)
print("is_winding_consistent:", mesh.is_winding_consistent)
print("volume:", mesh.volume)
print("center_mass:", mesh.center_mass)
print("bounds:\n", mesh.bounds)
