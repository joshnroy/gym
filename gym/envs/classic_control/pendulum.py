import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

ROD_TEST_COLOR = np.array([1., 0., 0.])
AXLE_TEST_COLOR = np.array([0., 1., 0.])
OFFSET_TEST_COLOR = np.array([0., 0., 0.])

class PendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g=10.0):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.g = g
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

        self.change_color()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,a):
        if a == 0:
            u = -0.1
        elif a == 1:
            u = 0.1
        elif a == 2:
            u = -0.3
        elif a == 3:
            u = 0.3
        elif a == 4:
            u = -0.5
        elif a == 5:
            u = 0.5
        elif a == 6:
            u = -0.7
        elif a == 7:
            u = 0.7
        elif a == 8:
            u = -1.0
        elif a == 9:
            u = 1.0
        elif a == 10:
            u = -1.3
        elif a == 11:
            u = 1.3
        elif a == 12:
            u = -1.6
        elif a == 13:
            u = 1.6
        elif a == 14:
            u = -2.0
        else:# a == 15:
            u = 2.0
        th, thdot = self.state # th := theta

        g = self.g
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)
        # u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])

        ret = np.concatenate([self.render().flatten(), [newthdot]])
        # return self._get_obs(), -costs, False, {}
        return ret, -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        # return self._get_obs()
        ret = np.concatenate([self.render().flatten(), [self.state[1]]])
        return ret

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='rgb_array'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(32,32)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            self.rod = rendering.make_capsule(1, .2)
            self.rod.set_color(self.rod_color[0], self.rod_color[1], self.rod_color[2])
            self.pole_transform = rendering.Transform()
            self.rod.add_attr(self.pole_transform)
            self.viewer.add_geom(self.rod)
            self.axle = rendering.make_circle(.05)
            self.axle.set_color(self.axle_color[0], self.axle_color[1], self.axle_color[2])
            self.viewer.add_geom(self.axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.rod.set_color(self.rod_color[0], self.rod_color[1], self.rod_color[2])
        self.axle.set_color(self.axle_color[0], self.axle_color[1], self.axle_color[2])

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def change_color(self):
        self.rod_color = np.random.random(3)
        while np.mean((self.rod_color - ROD_TEST_COLOR)**2) < 0.1:
            self.rod_color = np.random.random(3)

        self.axle_color = np.random.random(3)
        while np.mean((self.axle_color - AXLE_TEST_COLOR)**2) < 0.1:
            self.axle_color = np.random.random(3)

        self.offset_color = np.random.random(3)
        while np.mean((self.offset_color - OFFSET_TEST_COLOR)**2) < 0.1:
            self.offset_color = np.random.random(3)

    def change_color_test(self):
        self.rod_color = ROD_TEST_COLOR
        self.axle_color = ROD_AXLE_COLOR
        self.offset_color = OFFSET_TEST_COLOR

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
