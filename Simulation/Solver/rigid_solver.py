import taichi as ti
import numpy as np
import math
import time
import os
import sys
from typing import Tuple
from ..Base import BaseContainer

@ti.data_oriented
class RigidSolver():
    def __init__(self, container: BaseContainer, gravity: Tuple[float, float, float], dt):
        self.container = container
        self.cfg = container.cfg
        self.gravity = np.array(gravity)
        self.dt = dt

        self.current_time = 0.0
        self.current_step = 0

        # rigid objects and blocks
        self.current_objects = []
        self.current_blocks = []
        self.current_rigid = []
        self.rigid_objects = self.cfg.get_rigid_bodies()
        self.rigid_blocks = self.cfg.get_rigid_blocks()
        num_objects = len(self.rigid_objects)
        num_blocks = len(self.rigid_blocks)
        num_rigid = num_objects + num_blocks

        if num_rigid == 0:
            print("No rigid object or block is defined.")
        else:
            for rigid_objects in self.rigid_objects:
                self.init_rigid_object(rigid_objects)
            for rigid_blocks in self.rigid_blocks:
                self.init_rigid_block(rigid_blocks)

        self.rigid_particle_temp_positions = ti.Vector.field(3, dtype=ti.f32, shape=10)
        self.rigid_body_temp_centers_of_mass = ti.Vector.field(3, dtype=ti.f32, shape=10)
        self.rigid_body_temp_rotation_matrices = ti.Matrix.field(3, 3, dtype=ti.f32, shape=10)
        
    def init_rigid_object(self, rigid_object):
        index = rigid_object["objectID"]

        if index in self.current_rigid:
            return
        if rigid_object["entryTime"] > self.total_time:
            return
        
        self.current_objects.append(index)
        self.current_rigid.append(index)
        
        is_static = not rigid_object["isDynamic"]
        if is_static:
            return
        else:
            angle = rigid_object["rotationAngle"] / 360 * (2 * math.pi)
            direction = rigid_object["rotationAxis"]
            euler = np.array([direction[0] * angle, direction[1] * angle, direction[2] * angle])
            cx, cy, cz = np.cos(euler)
            sx, sy, sz = np.sin(euler)

            self.container.rigid_body_velocities[index] = np.array(rigid_object["velocity"], dtype=np.float32)
            self.container.rigid_body_angular_velocities[index] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.container.rigid_body_original_centers_of_mass[index] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.container.rigid_body_centers_of_mass[index] = np.array(rigid_object["translation"])
            self.container.rigid_body_rotations[index] = np.array([[cy * cz, -cy * sz, sy], [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy], [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy]])


    def init_rigid_block(self, rigid_block):
        raise NotImplementedError
    
    def apply_force(self, index, force: Tuple[float, float, float]):
        self.container.rigid_body_forces[index] += np.array(force)

    def apply_torque(self, index, torque: Tuple[float, float, float]):
        self.container.rigid_body_torques[index] += np.array(torque)

    def step(self):
        for container_id in range(self.container.object_num[None]):
            if self.container.rigid_body_is_dynamic[container_id] and self.container.object_materials[container_id] == self.container.material_rigid:
                force_i = self.container.rigid_body_forces[container_id]
                torque_i = self.container.rigid_body_torques[container_id]
                self.apply_force(container_id, force_i)
                self.apply_torque(container_id, torque_i)
                self.container.rigid_body_forces[container_id] = np.array([0.0, 0.0, 0.0])
                self.container.rigid_body_torques[container_id] = np.array([0.0, 0.0, 0.0])
                
        self.update_rigid_body_states()

    def update_rigid_body_states(self):
        for container_id in range(self.container.object_num[None]):
            if self.container.rigid_body_is_dynamic[container_id] and self.container.object_materials[container_id] == self.container.material_rigid:
                state_i = self.get_rigid_body_states(container_id)
                self.container.rigid_body_centers_of_mass[container_id] = state_i["position"]
                self.container.rigid_body_rotations[container_id] = state_i["rotation_matrix"]
                self.container.rigid_body_velocities[container_id] = state_i["linear_velocity"]
                self.container.rigid_body_angular_velocities[container_id] = state_i["angular_velocity"]

    def get_rigid_body_states(self, container_idx):
        linear_velocity = self.container.rigid_body_velocities[container_idx]
        angular_velocity = self.container.rigid_body_angular_velocities[container_idx]
        position = self.container.rigid_body_centers_of_mass[container_idx]
        rotation_matrix = self.container.rigid_body_rotations[container_idx]

        return {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "position": position,
            "rotation_matrix": rotation_matrix
        }

    @ti.kernel
    def update_rigid_velocities(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid and self.container.particle_is_dynamic[p_i]:
                self.container.particle_velocities[p_i] += self.dt[None] * self.container.particle_accelerations[p_i]

    @ti.kernel
    def update_rigid_positions(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid and self.container.particle_is_dynamic[p_i]:
                self.rigid_particle_temp_positions[p_i] = self.container.particle_positions[p_i] + self.dt[None] * self.container.particle_velocities[p_i]

    @ti.kernel
    def compute_temp_center_of_mass(self):
        self.rigid_body_temp_centers_of_mass.fill(0.0)

        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid and self.container.particle_is_dynamic[p_i]:
                object_id = self.container.particle_object_ids[p_i]
                self.rigid_body_temp_centers_of_mass[object_id] += self.container.V0 * self.container.particle_densities[p_i] * self.rigid_particle_temp_positions[p_i]

        for obj_i in range(self.container.object_num[None]):
            if self.container.rigid_body_is_dynamic[obj_i] and self.container.object_materials[obj_i] == self.container.material_rigid:
                self.rigid_body_temp_centers_of_mass[obj_i] /= self.container.rigid_body_masses[obj_i]

    @ti.kernel
    def solve_constraints(self):
        self.rigid_body_temp_rotation_matrices.fill(0.0)

        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid and self.container.particle_is_dynamic[p_i]:
                object_id = self.container.particle_object_ids[p_i]
                q = self.container.rigid_particle_original_positions[p_i] - self.container.rigid_body_original_centers_of_mass[object_id]
                p = self.rigid_particle_temp_positions[p_i] - self.rigid_body_temp_centers_of_mass[object_id]
                self.rigid_body_temp_rotation_matrices[object_id] += self.container.V0 * self.container.particle_densities[p_i] * p.outer_product(q)

        for obj_i in range(self.container.object_num[None]):
            if self.container.rigid_body_is_dynamic[obj_i] and self.container.object_materials[obj_i] == self.container.material_rigid:
                A_pq = self.rigid_body_temp_rotation_matrices[obj_i]
                R, S = ti.polar_decompose(A_pq)
                if all(abs(R) < 1e-6):
                    R = ti.Matrix.identity(ti.f32, 3)
                self.rigid_body_temp_rotation_matrices[obj_i] = R

        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid and self.container.particle_is_dynamic[p_i]:
                object_id = self.container.particle_object_ids[p_i]
                goal = self.rigid_body_temp_centers_of_mass[object_id] + self.rigid_body_temp_rotation_matrices[object_id] @ (self.container.rigid_particle_original_positions[p_i] - self.container.rigid_body_original_centers_of_mass[object_id])
                self.container.particle_positions[p_i] = goal
