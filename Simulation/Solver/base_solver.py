import taichi as ti
import numpy as np
import math
import time
import os
import sys
from ..Base import BaseContainer


@ti.data_oriented
class BaseSolver():
    def __init__(self, container: BaseContainer):
        self.container = container
        self.cfg = container.cfg

        # Gravity
        self.g = np.array(self.container.cfg.get_cfg("gravitation"))

        # density
        self.density = 1000.0
        self.density0 = self.container.cfg.get_cfg("density0")

        # surface tension
        if self.container.cfg.get_cfg("surface_tension"):
            self.surface_tension = self.container.cfg.get_cfg("surface_tension")
        else:
            self.surface_tension = 0.01

        # viscosity
        self.viscosity = self.container.cfg.get_cfg("viscosity")
        if self.container.cfg.get_cfg("viscosity_b"):
            self.viscosity_b = self.container.cfg.get_cfg("viscosity_b")
        else:
            self.viscosity_b = self.viscosity

        # time step
        self.dt = ti.field(float, shape=())
        self.dt[None] = 1e-4
        self.dt[None] = self.container.cfg.get_cfg("timeStepSize")

        
