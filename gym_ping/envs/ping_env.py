#!/usr/bin/env python3

# Certain parts made with help from:
#    http://pyode.sourceforge.net/tutorials/tutorial3.html

"""
Copyright (c) 2016 Dávid Kerekes - https://github.com/KerekesDavid/ping-env

 Permission is hereby granted, free of charge, to any person
 obtaining a copy of this software and associated documentation
 files (the "Software"), to deal in the Software without
 restriction, including without limitation the rights to use,
 copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following
 conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 OTHER DEALINGS IN THE SOFTWARE."""

from enum import Enum
import math

from gym import error, spaces
import gym

import ode

import numpy as np
import random
import time
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


# state enum

class PingEnvImg(gym.Env):

    metadata = {
        'render.modes': ['rgb_array', 'human']
    }

    class BallState(Enum):
        noneHit = 0
        farHit = 1
        nearHit = 2
        wallHit = 3
        batHit = 4
        nearSecondHit = 5
        farSecondHit = 6
        invalidHit = 7
        invalidBounce = 8

    def __init__(self):
        # OpenGL variables
        self.wp_height = 200
        self.wp_width = 200
        self.wp_x = 0
        self.wp_y = 0
        self.windowname = "PingEnvImg"
        self.window = None
        self.fov = 28
        self.cv1 = [0.2, 4.5, 0.001, 0.2, 1.0, 0, 0, 1, 0]
        self.cv2 = [3.5, 0.85, 0.0, 0.2, 1.0, 0, 0, 1, 0]

        # ODE variables
        self.world = ode.World()
        self.space = ode.Space()
        self.bodies = list()
        self.geoms = list()
        self.contactGroup = ode.JointGroup()

        # Simulation loop variables
        self.random = random.Random()
        self.lastseed = 12478
        self.random.seed(self.lastseed)
        self.dt = 0.02
        self.sim_div = 4
        self.counter = 0
        self.lasttime = time.time()
        self.lastBall = None
        self.hitState = self.BallState.noneHit
        self.prevHitState = self.hitState
        self._ldist = 0.0
        self._dist = 0.0

        self.world.setGravity((0, -9.81, 0))
        self.world.setERP(0.8)
        self.world.setCFM(1E-5)

        # Output variables
        self.rawdata = np.empty((self.wp_height, self.wp_width, 3), dtype='uint8')
        self.data = np.ones((self.wp_height, self.wp_width, 3), dtype='float16')
        self.observation = np.ones((self.wp_height, self.wp_width, 1), dtype='float16')
        self.databuf = np.ones_like(self.data)
        self.reward = 0
        self.done = False
        self.info = dict()
        # Observation space
        self.observation_space = spaces.Box(self.observation * -1.0, self.observation)

        # Valid bat bositions
        self.posClipMin = [0.6, 0.6, -1.0]
        self.posClipMax = [1.8, 1.0, 1.0]

        # Valid bat velocities
        self.action_space = spaces.Box(np.ones(6) * -1.5, np.ones(6) * 1.5)

        # Set up static objects
        #   Create table
        self.tableNet, self.tableFar, self.tableNear = self._create_table()

        #   Create planes to chatch balls
        self.hitWall = self._create_hit_wall()
        self.hitFloor = self._create_hit_floor()

        #   Create Bat
        self.bat = self._create_bat()

    # prepare openGL
    def _prepare_gl(self):
        """Prepare drawing.
        """

        # Viewport
        glViewport(0, 0, self.wp_width, self.wp_height)

        # Initialize
        glClearColor(0.8, 0.8, 0.9, 0)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_LIGHTING)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_FLAT)
        glEnable(GL_COLOR_MATERIAL)

        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, 1.0, 0.2, 20)

        # Initialize ModelView matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Light source
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 1, 0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1, 1, 1, 1])
        glEnable(GL_LIGHT0)

    # draw_body
    @staticmethod
    def _draw_body(body):
        """Draw an ODE body.
        """

        x, y, z = body.getPosition()
        r = body.getRotation()
        rot = [r[0], r[3], r[6], 0.,
               r[1], r[4], r[7], 0.,
               r[2], r[5], r[8], 0.,
               x, y, z, 1.0]
        glPushMatrix()
        glMultMatrixd(rot)
        if body.visible:
            glColor3fv(body.color)
            if body.shape == "Sphere":
                glutSolidSphere(body.radius, 24, 24)
            elif body.shape == "Box":
                sx, sy, sz = body.boxsize
                glScale(sx, sy, sz)
                glutSolidCube(1)
        glPopMatrix()

    # createBox
    def _create_box(self, density, lx, ly, lz):
        """Create a box body and its corresponding geom."""

        # Create body
        body = ode.Body(self.world)
        m = ode.Mass()
        m.setBox(density, lx, ly, lz)
        body.setMass(m)

        # Set parameters for drawing the body
        body.shape = "Box"
        body.boxsize = (lx, ly, lz)
        body.visible = True
        body.color = [0.2, 0.2, 0.5]

        # Create a box geom for collision detection
        geom = ode.GeomBox(self.space, lengths=body.boxsize)
        geom.setBody(body)
        return body, geom

    # createSphere
    def _create_sphere(self, density, rad):
        """Create a sphere body and its corresponding geom."""

        # Create body
        body = ode.Body(self.world)
        m = ode.Mass()
        m.setSphere(density, rad)
        body.setMass(m)

        # Set parameters for drawing the body
        body.shape = "Sphere"
        body.radius = rad
        body.visible = True
        body.color = [0.855, 0.471, 0.043]

        # Create a box geom for collision detection
        geom = ode.GeomSphere(self.space, body.radius)
        geom.setBody(body)

        return body, geom

    # drop_object
    def _drop_ball(self):
        """Drop an object into the scene."""

        body, geom = self._create_sphere(80.573, 0.02)  # 4cm atmero, 80.573 kg/m3 suruseg
        self.startPos = (self.random.gauss(-1.75, 0.1),
                         self.random.gauss(1.1, 0.1),
                         self.random.gauss(0, 0.4))
        body.setPosition(self.startPos)
        self.startVel = (self.random.gauss(7 - (2-self.startPos[1])*1.3, 0.3),
                         self.random.gauss(-1.4 * self.startPos[1], 0.1),
                         self.random.gauss(0, 0.5) - self.startPos[2]*1.5)
        body.setLinearVel(self.startVel)
        theta = self.random.uniform(0, 2 * math.pi)
        ct = math.cos(theta)
        st = math.sin(theta)
        body.setRotation([ct, 0., -st, 0., 1., 0., st, 0., ct])
        self.bodies.append(body)
        self.geoms.append(geom)
        self.counter = 0
        return geom

    def _delete_obj(self, obj):
        """Remove an object from the scene"""

        if obj is not None:
            self.bodies.remove(obj.getBody())
            self.geoms.remove(obj)

    def _create_table(self):

        zero_rot = np.full((9, 1), np.finfo('float32').eps, dtype='float32')
        zero_rot[4] = 1.0

        # Near table
        body1, geom1 = self._create_box(1000, 2.74 / 2.0, 0.06, 1.525)  # asztal hivatalos meretei
        body1.setPosition([2.74/4, 0.76, 0])  # 0.76m hivatalos magassag
        body1.setRotation(zero_rot)
        body1.setKinematic()
        self.bodies.append(body1)
        self.geoms.append(geom1)

        # Far table
        body2, geom2 = self._create_box(1000, 2.74 / 2.0, 0.06, 1.525)  # asztal hivatalos meretei
        body2.setPosition([-2.74/4, 0.76, 0])  # 0.76m hivatalos magassag
        body2.setRotation(zero_rot)
        body2.setKinematic()
        self.bodies.append(body2)
        self.geoms.append(geom2)

        # Net
        body3, geom3 = self._create_box(1000, 0.01, 0.1545, 1.525)  # halo
        body3.setPosition([0, 0.76 + 0.03 + 0.1545/2, 0])  # 0.76m hivatalos magassag
        body3.setRotation(zero_rot)
        body3.setKinematic()
        self.bodies.append(body3)
        self.geoms.append(geom3)

        return geom3, geom2, geom1

    def _create_hit_wall(self):

        body, geom = self._create_box(100, 100, 0.01, 10)
        body.setPosition([1.8, 0.76, 0])
        body.setRotation([0, -1, 0, 1, 0, 0, 0, 0, 1])
        body.setKinematic()
        body.visible = False
        self.bodies.append(body)
        self.geoms.append(geom)

        return geom

    def _create_hit_floor(self):

        zero_rot = np.full((9, 1), np.finfo('float32').eps, dtype='float32')
        zero_rot[4] = 1.0

        body, geom = self._create_box(1000, 1000, 0.01, 10)
        body.setPosition([0.0, 0.40, 0])
        body.setRotation(zero_rot)
        body.setKinematic()
        body.visible = False
        self.bodies.append(body)
        self.geoms.append(geom)

        return geom

    def _create_bat(self):

        body, geom = self._create_box(1000, 0.158, 0.01, 0.150)
        body.setPosition([1.5, 0.8, 0.0])
        body.setRotation([0, -1, 0, 1, 0, 0, 0, 0, 1])  # TODO and Gaussian rotaion
        body.setKinematic()
        body.visible = True
        body.color = [0.8, 0.2, 0.2]
        self.bodies.append(body)
        self.geoms.append(geom)

        return geom

    # Collision callback
    @staticmethod
    def _near_callback(self, geom1, geom2):
        """Callback function for the collide() method.

        This function checks if the given self.geoms do collide and
        creates contact joints if they do.
        """

        # Set of geoms
        gset = {geom1, geom2}

        # Check if the objects do collide
        contacts = ode.collide(geom1, geom2)

        # Create contact joints
        for c in contacts:
            c.setBounce(0.77)  # standard bounce pingpong labda + asztal kozott
            c.setMu(0.25)  # approx mu
            j = ode.ContactJoint(self.world, self.contactGroup, c)
            j.attach(geom1.getBody(), geom2.getBody())
            if {self.lastBall} <= gset:
                if not {self.tableNet, self.hitFloor}.isdisjoint(gset):
                        self.hitState = self.BallState.invalidBounce
                elif {self.hitWall} <= gset:
                    if self.hitState == self.BallState.nearHit:
                        self.hitState = self.BallState.wallHit
                    else:
                        self.hitState = self.BallState.invalidBounce
                elif {self.tableFar} <= gset:
                    if self.hitState == self.BallState.noneHit:
                        self.hitState = self.BallState.farHit
                    elif self.hitState == self.BallState.batHit:
                        self.hitState = self.BallState.farSecondHit
                    else:
                        self.hitState = self.BallState.invalidBounce
                elif {self.tableNear} <= gset:
                    if self.hitState == self.BallState.farHit:
                        self.hitState = self.BallState.nearHit
                    elif self.hitState == self.BallState.batHit:
                        self.hitState = self.BallState.nearSecondHit
                    else:
                        self.hitState = self.BallState.invalidBounce
                elif {self.bat} <= gset:
                    if self.hitState == self.BallState.nearHit:
                        self.hitState = self.BallState.batHit
                    else:
                        self.hitState = self.BallState.invalidHit
            elif {self.bat, self.tableNear} == gset:
                self.hitState = self.BallState.invalidHit
            # TODO else:
            #     print('Optimize me out!')

    def _drag(self):

        fa = np.array(self.lastBall.getBody().getLinearVel())
        n = fa.dot(fa)
        d = 1/2*1.225*0.6*0.00125663706*n  # 1/2 * air density * drag coefficient * area * velocity^2
        n = np.sqrt(n)
        if n > 0.00001:
            fa *= d/n
            self.lastBall.getBody().addForce((-fa[0], -fa[1], -fa[2]))

    def _sim_step(self, action):
        self.counter += 1
        self.prevHitState = self.hitState

        self.bat.getBody().setLinearVel(action[0:3])
        self.bat.getBody().setAngularVel(action[3:6])

        # Simulate
        for i in range(self.sim_div):
            # Add drag
            self._drag()

            # Detect collisions and create contact joints
            self.space.collide(self, self._near_callback)

            # Simulation step
            self.world.step(self.dt / self.sim_div)

            # Remove all contact joints
            self.contactGroup.empty()

    def _reset_sim(self):
        self._delete_obj(self.lastBall)
        self.lastBall = self._drop_ball()
        self._delete_obj(self.bat)
        self.bat = self._create_bat()
        self.hitState = self.BallState.noneHit
        self.counter = 0

    def _validate_throw(self):
        """
        Run until the throw is valid
        Resets the ball and state to beginning of the throw
        """

        while True:
            saved_state = self.random.getstate()
            self._reset_sim()

            while self.counter < 100:
                self._sim_step([0, 0, 0, 0, 0, 0])

                if self.prevHitState == self.BallState.nearHit and \
                        (self.hitState == self.BallState.wallHit or self.hitState == self.BallState.batHit):
                    break
            else:           # This is ugly but cool. it exits the outer loop only if the inner was broken
                continue
            break

        self.random.setstate(saved_state)  # In a language without dowhile, how can you flag this as unassinged dear PyCharm?
        self._reset_sim()

    def _set_after_hit_reward(self):
        while self.counter < 100:
            self._sim_step([0, 0, 0, 0, 0, 0])

            if self.hitState == self.BallState.farSecondHit:
                self.reward += 19.0
                break
            elif self.hitState == self.BallState.nearSecondHit:
                # reward hitting (bouncing) back the ball further
                p1 = np.array(self.lastBall.getBody().getPosition())
                self.reward += 1.0 + min(max(1.4 - p1[0], 0.0), 1.5)
                break
            elif self.hitState == self.BallState.invalidHit or self.hitState == self.BallState.invalidBounce:
                break

    def _draw_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Render stereo image upper part
        glViewport(0, int(self.wp_height / 2), self.wp_width, int(self.wp_height / 2))
        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, int(self.wp_width / (self.wp_height / 2)), 0.2, 20)
        # Initialize ModelView matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Cam 1
        gluLookAt(self.cv1[0], self.cv1[1], self.cv1[2],
                  self.cv1[3], self.cv1[4], self.cv1[5],
                  self.cv1[6], self.cv1[7], self.cv1[8])
        for b in self.bodies:
            self._draw_body(b)

        # Render stereo image lower part
        glViewport(0, 0, self.wp_width, int(self.wp_height / 2))
        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, int(self.wp_width / (self.wp_height / 2)), 0.2, 20)
        # Initialize ModelView matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Cam 2
        gluLookAt(self.cv2[0], self.cv2[1], self.cv2[2],
                  self.cv2[3], self.cv2[4], self.cv2[5],
                  self.cv2[6], self.cv2[7], self.cv2[8])
        for b in self.bodies:
            self._draw_body(b)

        glutSwapBuffers()
        glutPostRedisplay()
        glutMainLoopEvent()
        # Get info
        glReadBuffer(GL_FRONT)

    def _step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step()
                            calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # Assert action size
        action = np.array(action)
        if not all(np.logical_and(self.action_space.low <= action, action <= self.action_space.high)):
            print('Action is out of bounds, clipping: {0}'.format(action))
            action = np.clip(action, self.action_space.low, self.action_space.high)

        # Step the simulation
        self._sim_step(action)

        # Draw the scene
        self._draw_scene()

        self.rawdata = glReadPixels(0, 0, self.wp_width, self.wp_height,
                                    GL_RGB, GL_UNSIGNED_BYTE, outputType=None)
        self.data = np.flipud(self.rawdata.reshape((self.wp_height, self.wp_width, 3), order='C')) / 255.0
        self.observation = np.amax(self.databuf - self.data*0.9, -1, keepdims=True)  # Take maxdif but leave channel dim
        self.databuf = self.data

        # fill in some info
        self.info['hitState'] = self.hitState

        # Set rewards
        self.reward = 0.0

        p1 = np.array(self.lastBall.getBody().getPosition())
        p2 = np.array(self.bat.getBody().getPosition())
        self._dist = self._ldist
        self._ldist = np.linalg.norm(p1[1:3] - p2[1:3])

        # Check if paddle is out of bounds
        batpos = np.array(self.bat.getBody().getPosition())
        if np.all(np.logical_or(batpos <= np.array(self.posClipMin), batpos >= np.array(self.posClipMax))):
            self.reward -= 1.0
            self.done = True
            self.info['faliure'] = "Bat out of bounds"

        # Check for invalid state
        if self.hitState == self.BallState.wallHit:
            self.reward -= self._dist
            self.done = True
        elif self.hitState == self.BallState.invalidHit:
            self.reward -= 1.0
            self.done = True
        elif self.hitState == self.BallState.invalidBounce or self.hitState == self.BallState.nearSecondHit:
            self.done = True

        # Check for succesful hits
        if self.hitState == self.BallState.batHit and self.prevHitState != self.BallState.batHit:
            self.reward += 1.0
            # Calculate further rewards based on bounces
            self._set_after_hit_reward()
            self.done = True
        else:
            # Give reward based on change of distance from bat
            # self.reward += (self._ldist - self._dist)

            # Stop sim after overshoot
            if (p1 - p2)[0] > 0.1:
                # Give reward based on distance from bat
                self.reward -= self._dist
                self.done = True

        return self.observation.copy(), self.reward, self.done, self.info

    def _reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        # Initialize GLUT if neccessary
        if self.window is None:
            glutInit([])
            # Open a window
            glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE)
            glutInitWindowPosition(self.wp_x, self.wp_y)
            glutInitWindowSize(self.wp_width, self.wp_height)
            self.window = glutCreateWindow(self.windowname)
            self._prepare_gl()
            glutMainLoopEvent()

        # Reset output vars
        self.reward = 0
        self.done = False
        self.info = dict()
        # Reset internal vars
        self.databuf = np.ones_like(self.data)
        # Find valid starting pos
        self._validate_throw()
        # Throw first step out to get consistent observations
        self._step(np.zeros(self.action_space.shape))
        self._step(np.zeros(self.action_space.shape))

        return self.observation.copy()

    def _close(self):
        """
            Environments will automatically close() themselves when
            garbage collected or when the program exits.
        """
        self.done = True

        if self.window is not None:
            glutHideWindow(self.window)
            glutDestroyWindow(self.window)
            self.window = None

    def _seed(self, seed=None):
        self.lastseed = seed
        self.random.seed(seed)

    def _render(self, mode='rgb_array', close=False):

        if mode == 'rgb_array':
            return self.rawdata
        else:
            pass


class PingEnvPos(gym.Env):

    metadata = {
        'render.modes': ['rgb_array', 'human']
    }

    class BallState(Enum):
        noneHit = 0
        farHit = 1
        nearHit = 2
        wallHit = 3
        batHit = 4
        nearSecondHit = 5
        farSecondHit = 6
        invalidHit = 7

    def __init__(self):
        # OpenGL variables
        self.wp_height = 200
        self.wp_width = 200
        self.wp_x = 0
        self.wp_y = 0
        self.windowname = "PingEnv"
        self.window = None
        self.fov = 28
        self.cv1 = [0.2, 5.0, 1.0, 0.2, 1.0, 0, 0, 1, 0]
        self.cv2 = [3.0, 1.0, 0.0, 0.2, 1.0, 0, 0, 1, 0]

        # ODE variables
        self.world = ode.World()
        self.space = ode.Space()
        self.bodies = list()
        self.geoms = list()
        self.contactGroup = ode.JointGroup()

        # Simulation loop variables
        self.random = random.Random()
        self.lastseed = 12478
        self.random.seed(self.lastseed)
        self.dt = 0.02
        self.sim_div = 4
        self.counter = 0
        self.lasttime = time.time()
        self.lastBall = None
        self.hitState = self.BallState.noneHit
        self.prevHitState = self.hitState
        self._ldist = 0.0
        self._dist = 0.0

        self.world.setGravity((0, -9.81, 0))
        self.world.setERP(0.8)
        self.world.setCFM(1E-5)

        # Output variables
        self.observation = np.ones((9, ), dtype='float32')  # 3*vec3
        self.reward = 0
        self.done = False
        self.info = dict()
        # Observation space
        self.observation_space = spaces.Box(self.observation * -10.0, self.observation * 10.0)

        # Valid bat bositions
        self.posClipMin = [0.4, 0.5, -1.0]
        self.posClipMax = [1.7, 1.7, 1.0]

        # Valid bat velocities
        self.action_space = spaces.Box(np.ones(3) * -0.5, np.ones(3) * 0.5)

        # Set up static objects
        #   Create table
        self.tableFar, self.tableNear = self._create_table()

        #   Create planes to chatch balls
        self.hitWall = self._create_hit_wall()

        #   Create Bat
        self.bat = self._create_bat()

    # prepare openGL
    def _prepare_gl(self):
        """Prepare drawing.
        """

        # Viewport
        glViewport(0, 0, self.wp_width, self.wp_height)

        # Initialize
        glClearColor(0.8, 0.8, 0.9, 0)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_LIGHTING)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_FLAT)
        glEnable(GL_COLOR_MATERIAL)

        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, 1.0, 0.2, 20)

        # Initialize ModelView matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Light source
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 1, 0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1, 1, 1, 1])
        glEnable(GL_LIGHT0)

    # draw_body
    @staticmethod
    def _draw_body(body):
        """Draw an ODE body.
        """

        x, y, z = body.getPosition()
        r = body.getRotation()
        rot = [r[0], r[3], r[6], 0.,
               r[1], r[4], r[7], 0.,
               r[2], r[5], r[8], 0.,
               x, y, z, 1.0]
        glPushMatrix()
        glMultMatrixd(rot)
        if body.visible:
            glColor3fv(body.color)
            if body.shape == "Sphere":
                glutSolidSphere(body.radius, 24, 24)
            elif body.shape == "Box":
                sx, sy, sz = body.boxsize
                glScale(sx, sy, sz)
                glutSolidCube(1)
        glPopMatrix()

    # createBox
    def _create_box(self, density, lx, ly, lz):
        """Create a box body and its corresponding geom."""

        # Create body
        body = ode.Body(self.world)
        m = ode.Mass()
        m.setBox(density, lx, ly, lz)
        body.setMass(m)

        # Set parameters for drawing the body
        body.shape = "Box"
        body.boxsize = (lx, ly, lz)
        body.visible = True
        body.color = [0.2, 0.2, 0.5]

        # Create a box geom for collision detection
        geom = ode.GeomBox(self.space, lengths=body.boxsize)
        geom.setBody(body)
        return body, geom

    # createSphere
    def _create_sphere(self, density, rad):
        """Create a sphere body and its corresponding geom."""

        # Create body
        body = ode.Body(self.world)
        m = ode.Mass()
        m.setSphere(density, rad)
        body.setMass(m)

        # Set parameters for drawing the body
        body.shape = "Sphere"
        body.radius = rad
        body.visible = True
        body.color = [0.855, 0.471, 0.043]

        # Create a box geom for collision detection
        geom = ode.GeomSphere(self.space, body.radius)
        geom.setBody(body)

        return body, geom

    # drop_object
    def _drop_ball(self):
        """Drop an object into the scene."""

        body, geom = self._create_sphere(80.573, 0.02)  # 4cm atmero, 80.573 kg/m3 suruseg
        self.startPos = (self.random.gauss(-1.75, 0.1),
                         self.random.gauss(1.1, 0.1),
                         self.random.gauss(0, 0.4))
        body.setPosition(self.startPos)
        self.startVel = (self.random.gauss(7 - (2-self.startPos[1])*1.3, 0.3),
                         self.random.gauss(-1.4 * self.startPos[1], 0.1),
                         self.random.gauss(0, 0.5) - self.startPos[2]*1.5)
        body.setLinearVel(self.startVel)
        theta = self.random.uniform(0, 2 * pi)
        ct = cos(theta)
        st = sin(theta)
        body.setRotation([ct, 0., -st, 0., 1., 0., st, 0., ct])
        self.bodies.append(body)
        self.geoms.append(geom)
        self.counter = 0
        return geom

    def _delete_obj(self, obj):
        """Remove an object from the scene"""

        if obj is not None:
            self.bodies.remove(obj.getBody())
            self.geoms.remove(obj)

    def _create_table(self):

        zero_rot = np.full((9, 1), np.finfo('float32').eps, dtype='float32')
        zero_rot[4] = 1.0

        body1, geom1 = self._create_box(1000, 2.74 / 2.0, 0.06, 1.525)  # asztal hivatalos meretei
        body1.setPosition([2.74/4, 0.76, 0])  # 0.76m hivatalos magassag
        body1.setRotation(zero_rot)
        body1.setKinematic()
        self.bodies.append(body1)
        self.geoms.append(geom1)

        body2, geom2 = self._create_box(1000, 2.74 / 2.0, 0.06, 1.525)  # asztal hivatalos meretei
        body2.setPosition([-2.74/4, 0.76, 0])  # 0.76m hivatalos magassag
        body2.setRotation(zero_rot)
        body2.setKinematic()
        self.bodies.append(body2)
        self.geoms.append(geom2)

        body3, geom3 = self._create_box(1000, 0.01, 0.1545, 1.525)  # halo
        body3.setPosition([0, 0.76 + 0.03 + 0.1545/2, 0])  # 0.76m hivatalos magassag
        body3.setRotation(zero_rot)
        body3.setKinematic()
        self.bodies.append(body3)
        self.geoms.append(geom3)

        return geom2, geom1

    def _create_hit_wall(self):

        body, geom = self._create_box(1000, 10, 0.01, 10)
        body.setPosition([2.5, 0.76, 0])
        body.setRotation([0, -1, 0, 1, 0, 0, 0, 0, 1])
        body.setKinematic()
        body.visible = False
        self.bodies.append(body)
        self.geoms.append(geom)

        return geom

    def _create_bat(self):

        body, geom = self._create_box(1000, 0.158, 0.01, 0.150)
        body.setPosition([1.5, 0.7, 0.0])
        body.setRotation([0, -1, 0, 1, 0, 0, 0, 0, 1])  # TODO and Gaussian rotaion
        body.setKinematic()
        body.visible = True
        body.color = [0.8, 0.2, 0.2]
        self.bodies.append(body)
        self.geoms.append(geom)

        return geom

    # Collision callback
    @staticmethod
    def _near_callback(self, geom1, geom2):
        """Callback function for the collide() method.

        This function checks if the given self.geoms do collide and
        creates contact joints if they do.
        """

        # Set of geoms
        gset = {geom1, geom2}

        # Check if the objects do collide
        contacts = ode.collide(geom1, geom2)

        # Create contact joints
        for c in contacts:
            c.setBounce(0.77)  # standard bounce pingpong labda + asztal kozott
            c.setMu(0.25)  # approx mu
            j = ode.ContactJoint(self.world, self.contactGroup, c)
            j.attach(geom1.getBody(), geom2.getBody())
            if {self.lastBall} <= gset:
                if {self.hitWall} <= gset:
                    if self.hitState == self.BallState.nearHit:
                        self.hitState = self.BallState.wallHit
                elif {self.tableFar} <= gset:
                    if self.hitState == self.BallState.noneHit:
                        self.hitState = self.BallState.farHit
                    elif self.hitState == self.BallState.nearSecondHit:
                        self.hitState = self.BallState.farSecondHit
                    else:
                        self.hitState = self.BallState.invalidHit
                elif {self.tableNear} <= gset:
                    if self.hitState == self.BallState.farHit:
                        self.hitState = self.BallState.nearHit
                    elif self.hitState == self.BallState.batHit:
                        self.hitState = self.BallState.nearSecondHit
                    else:
                        self.hitState = self.BallState.invalidHit
                elif {self.bat} <= gset:
                    if self.hitState == self.BallState.nearHit:
                        self.hitState = self.BallState.batHit
                    else:
                        self.hitState = self.BallState.invalidHit
            # TODO else:
            #     print('Optimize me out!')

    def _drag(self):

        fa = np.array(self.lastBall.getBody().getLinearVel())
        n = fa.dot(fa)
        d = 1/2*1.225*0.6*0.00125663706*n  # 1/2 * air density * drag coefficient * area * velocity^2
        n = np.sqrt(n)
        if n > 0.00001:
            fa *= d/n
            self.lastBall.getBody().addForce((-fa[0], -fa[1], -fa[2]))

    def _sim_step(self, action):
        self.counter += 1
        self.prevHitState = self.hitState

        self.bat.getBody().setLinearVel(action[0:3])
        # self.bat.getBody().setAngularVel(action[3:6])

        # Simulate
        for i in range(self.sim_div):
            # Add drag
            self._drag()

            # Detect collisions and create contact joints
            self.space.collide(self, self._near_callback)

            # Simulation step
            self.world.step(self.dt / self.sim_div)

            # Remove all contact joints
            self.contactGroup.empty()

    def _reset_sim(self):
        self._delete_obj(self.lastBall)
        self.lastBall = self._drop_ball()
        self._delete_obj(self.bat)
        self.bat = self._create_bat()
        self.hitState = self.BallState.noneHit
        self.counter = 0

    def _validate_throw(self):
        """
        Run until the throw is valid
        Resets the ball and state to beginning of the throw
        """

        while True:
            saved_state = self.random.getstate()
            self._reset_sim()

            while self.counter < 100:
                self._sim_step([0, 0, 0, 0, 0, 0])
                if self.prevHitState == self.BallState.nearHit and \
                        (self.hitState == self.BallState.wallHit or self.hitState == self.BallState.batHit):
                    break
            else:           # This is ugly but cool. it exits the outer loop only if the inner was broken
                continue
            break

        self.random.setstate(saved_state)  # In a language without dowhile, how can you flag this as unassinged dear PyCharm?
        self._reset_sim()

    def _draw_scene(self):
        # Render stereo image upper part
        glViewport(0, int(self.wp_height / 2), self.wp_width, int(self.wp_height / 2))
        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, int(self.wp_width / (self.wp_height / 2)), 0.2, 20)
        # Initialize ModelView matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Cam 1
        gluLookAt(self.cv1[0], self.cv1[1], self.cv1[2],
                  self.cv1[3], self.cv1[4], self.cv1[5],
                  self.cv1[6], self.cv1[7], self.cv1[8])
        for b in self.bodies:
            self._draw_body(b)

        # Render stereo image lower part
        glViewport(0, 0, self.wp_width, int(self.wp_height / 2))
        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, int(self.wp_width / (self.wp_height / 2)), 0.2, 20)
        # Initialize ModelView matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Cam 2
        gluLookAt(self.cv2[0], self.cv2[1], self.cv2[2],
                  self.cv2[3], self.cv2[4], self.cv2[5],
                  self.cv2[6], self.cv2[7], self.cv2[8])
        for b in self.bodies:
            self._draw_body(b)

    def _step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step()
                            calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # Assert action size
        action = np.array(action)
        assert all(np.logical_and(self.action_space.low <= action, action <= self.action_space.high))

        # Step the simulation
        self._sim_step(action)

        self.reward = 0.0

        p1 = np.array(self.lastBall.getBody().getPosition())
        p2 = np.array(self.bat.getBody().getPosition())
        self._dist = self._ldist
        self._ldist = np.linalg.norm(p1[1:3] - p2[1:3])

        # Check if paddle is out of bounds
        batpos = np.array(self.bat.getBody().getPosition())
        if np.all(np.logical_or(batpos <= np.array(self.posClipMin), batpos >= np.array(self.posClipMax))):
            self.reward -= 1.0
            self.done = True
            self.info['faliure'] = "Bat out of bounds"

        # Check for succesful hits
        if self.hitState == self.BallState.batHit and self.prevHitState != self.BallState.batHit:
            self.reward += 1.0
            self.done = True
        else:
            # Give reward based on change of distance from bat
            # self.reward += (self._ldist - self._dist)*10

            # Stop sim after overshoot
            if (p1 - p2)[0] > 0.1:
                # Give reward based on distance from bat
                self.reward -= self._dist
                self.done = True

        # Check for invalid state
        if self.hitState == self.BallState.wallHit:
            self.reward -= self._dist
            self.done = True
        elif self.hitState == self.BallState.invalidHit:
            self.reward -= self._dist
            self.done = True

        self.observation[0:3] = np.reshape(self.lastBall.getBody().getPosition(), (3, ))
        self.observation[3:6] = np.reshape(self.lastBall.getBody().getLinearVel(), (3, ))
        self.observation[6:9] = np.reshape(self.bat.getBody().getPosition(), (3, ))
        # self.observation[9:13] = np.reshape(self.bat.getBody().getQuaternion(), (4, ))

        # fill in some info
        self.info['hitState'] = self.hitState

        return self.observation.copy(), self.reward, self.done, self.info

    def _reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        # Initialize GLUT if neccessary
        if self.window is None:
            glutInit([])
            # Open a window
            glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE)
            glutInitWindowPosition(self.wp_x, self.wp_y)
            glutInitWindowSize(self.wp_width, self.wp_height)
            self.window = glutCreateWindow(self.windowname)
            self._prepare_gl()
            glutMainLoopEvent()

        # Reset output vars
        self.reward = 0
        self.done = False
        self.info = dict()
        # Find valid starting pos
        self._validate_throw()
        # Throw first step out to get consistent observations
        self._step(np.zeros(self.action_space.shape))
        self._step(np.zeros(self.action_space.shape))

        return self.observation.copy()

    def _close(self):
        """
            Environments will automatically close() themselves when
            garbage collected or when the program exits.
        """
        self.done = True

        if self.window is not None:
            glutHideWindow(self.window)
            glutDestroyWindow(self.window)
            self.window = None

    def _seed(self, seed=None):
        self.lastseed = seed
        self.random.seed(seed)

    def _render(self, mode='rgb_array', close=False):

        # Draw the scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._draw_scene()
        glutSwapBuffers()
        glutPostRedisplay()
        glutMainLoopEvent()
        # Get info
        glReadBuffer(GL_FRONT)

        if mode == 'rgb_array':
            self.rawdata = glReadPixels(0, 0, self.wp_width, self.wp_height,
                                        GL_RGB, GL_UNSIGNED_BYTE, outputType=None)
            return self.rawdata
