import os
import numpy as np
import trimesh


def generate_hill_mesh(
    seed: int,
    x_min: float = -12.0,
    x_max: float = 42.0,
    y_min: float = -12.0,
    y_max: float = 12.0,
    nx: int = 80,
    ny: int = 40,
    max_height: float = 4.5,
) -> trimesh.Trimesh:
    """Generate a random smooth hilly terrain mesh with flat spawn/goal zones."""
    rng = np.random.default_rng(seed)

    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")

    # Sum of random 2D sinusoids for smooth hills and ramps
    z = np.zeros_like(grid_x)
    num_waves = 5
    for _ in range(num_waves):
        amp = rng.uniform(0.3, 1.0)
        fx = rng.uniform(0.08, 0.30)
        fy = rng.uniform(0.08, 0.30)
        px = rng.uniform(0.0, 2.0 * np.pi)
        py = rng.uniform(0.0, 2.0 * np.pi)
        z += amp * np.sin(fx * grid_x + px) * np.cos(fy * grid_y + py)

    # Shift so minimum is 0 and normalize to [0, max_height]
    z = z - z.min()
    if z.max() > 1e-6:
        z = z * (rng.uniform(0.5, 1.0) * max_height / z.max())

    # Smoothly taper height to 0 at spawn (X < -2) and goal (X > 35)
    taper = np.ones_like(grid_x)
    left_mask = (grid_x > -4.0) & (grid_x < 2.0)
    taper[grid_x <= -4.0] = 0.0
    taper[left_mask] = 0.5 * (1.0 - np.cos(np.pi * (grid_x[left_mask] + 4.0) / 6.0))

    right_mask = (grid_x > 31.0) & (grid_x < 36.0)
    taper[grid_x >= 36.0] = 0.0
    taper[right_mask] = 0.5 * (1.0 + np.cos(np.pi * (grid_x[right_mask] - 31.0) / 5.0))

    z *= taper

    vertices = np.column_stack([grid_x.ravel(), grid_y.ravel(), z.ravel()])

    # Build triangle faces for the grid
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v0 = i * ny + j
            v1 = (i + 1) * ny + j
            v2 = (i + 1) * ny + (j + 1)
            v3 = i * ny + (j + 1)
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)
    return mesh


def write_urdf(urdf_path: str, obj_filename: str, name: str) -> None:
    urdf_content = f"""<?xml version='1.0' encoding='UTF-8'?>
<robot name="{name}">
  <link name="terrain_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual name="terrain_visual">
      <geometry>
        <mesh filename="{obj_filename}" scale="1 1 1"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </visual>
    <collision name="terrain_collision">
      <geometry>
        <mesh filename="{obj_filename}" scale="1 1 1"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </collision>
  </link>
</robot>
"""
    with open(urdf_path, "w") as f:
        f.write(urdf_content)


def main(num_terrains: int = 20) -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for idx in range(num_terrains):
        obj_name = f"hill_{idx}.obj"
        urdf_name = f"hill_{idx}.urdf"
        mesh = generate_hill_mesh(seed=idx)
        mesh.export(os.path.join(out_dir, obj_name))
        write_urdf(os.path.join(out_dir, urdf_name), obj_name, f"hill_{idx}")
    print(f"Generated {num_terrains} random hill terrains in {out_dir}")


if __name__ == "__main__":
    main(num_terrains=20)
