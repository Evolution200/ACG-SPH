import taichi as ti
import numpy as np
import math
from typing import Tuple
from ..Base import BaseContainer

@ti.data_oriented
class RigidSolver():
    def __init__(self, container: BaseContainer, gravity: Tuple[float, float, float], dt):
        self.container = container
        self.cfg = container.cfg
        self.gravity = np.array(gravity)
        self.dt = dt
        self.rigid_body_scales = ti.Vector.field(3, dtype=ti.f32, shape=10)

        self.current_time = 0.0
        self.current_step = 0

        # rigid objects
        self.current_objects = []
        self.rigid_objects = self.cfg.get_rigid_bodies()
        num_objects = len(self.rigid_objects)

        if num_objects == 0:
            print("No rigid object is defined.")
        else:
            for rigid_objects in self.rigid_objects:
                self.init_rigid_object(rigid_objects)

        self.rigid_particle_temp_positions = ti.Vector.field(3, dtype=ti.f32, shape=10)
        self.rigid_body_temp_centers_of_mass = ti.Vector.field(3, dtype=ti.f32, shape=10)
        self.rigid_body_temp_rotation_matrices = ti.Matrix.field(3, 3, dtype=ti.f32, shape=10)
        
    def init_rigid_object(self, rigid_object):
        index = rigid_object["objectID"]

        if index in self.current_objects:
            return
        if rigid_object["entryTime"] > self.total_time:
            return
        
        self.current_objects.append(index)
        
        is_dynamic = not rigid_object["isDynamic"]
        if not is_dynamic:
            return
        else:
            angle = rigid_object["rotationAngle"] / 360 * (2 * math.pi)
            direction = rigid_object["rotationAxis"]
            euler = np.array([direction[0] * angle, direction[1] * angle, direction[2] * angle])
            cx, cy, cz = np.cos(euler)
            sx, sy, sz = np.sin(euler)

            self.rigid_body_scales[index] = np.array(rigid_object["scale"], dtype=np.float32)
            self.container.rigid_body_velocities[index] = np.array(rigid_object["velocity"], dtype=np.float32)
            self.container.rigid_body_angular_velocities[index] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.container.rigid_body_original_centers_of_mass[index] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.container.rigid_body_centers_of_mass[index] = np.array(rigid_object["translation"])
            self.container.rigid_body_rotations[index] = np.array([[cy * cz, -cy * sz, sy], [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy], [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy]])

    def create_boundary(self, thickness: float = 0.01):
        # we do not want the rigid to hit the boundary of fluid so we move each wall inside a little bit.
        eps = self.container.particle_diameter + self.container.domain_box_thickness 
        domain_start = self.container.domain_start
        domain_end = self.container.domain_end

        self.start = np.array(domain_start) + eps
        self.end = np.array(domain_end) - eps

    def update_velocity(self, index):
        self.container.rigid_body_velocities[index] += self.gravity * self.dt + self.container.rigid_body_forces[index] / self.container.rigid_body_masses[index] * self.dt

    def update_angular_velocity(self, index):
        self.container.rigid_body_angular_velocities[index] += self.container.rigid_body_torques[index] / (0.4 * self.container.rigid_body_masses[index] * self.rigid_body_scales[index].norm() ** 2) * self.dt

    def update_position(self, index):
        self.container.rigid_body_centers_of_mass[index] += self.container.rigid_body_velocities[index] * self.dt
        rotation_vector = self.container.rigid_body_angular_velocities[index] * self.dt
        theta = rotation_vector.norm()
        omega = rotation_vector.normalized()
        omega_cross = np.array([[0.0, -omega[2], omega[1]], [omega[2], 0.0, -omega[0]], [-omega[1], omega[0], 0.0]])
        rotation_matrix = np.eye(3) + np.sin(theta) * omega_cross + (1 - np.cos(theta)) * omega_cross @ omega_cross
        self.container.rigid_body_rotations[index] = rotation_matrix @ self.container.rigid_body_rotations[index]

    def step(self):

        for index in range(self.container.object_num[None]):
            self.update_velocity(index)
            self.update_angular_velocity(index)
            self.update_position(index)
            self.container.rigid_body_forces[index] = np.array([0.0, 0.0, 0.0])
            self.container.rigid_body_torques[index] = np.array([0.0, 0.0, 0.0])