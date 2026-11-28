"""One-step suction-pick env: obs = top-down depth + valid mask, action = one cell of 128x128. No torch."""
import gymnasium as gym
import mujoco
import numpy as np

from cpick.seeds import SPLITS
from cpick.sim.camera import Camera
from cpick.sim.geometry import backproject
from cpick.sim.reset import CanDropper
from cpick.sim.scene_builder import build_scene_xml
from cpick.sim.suction import SuctionChecker

GRID = 128          # action cell (i, j) -> pixel (2j, 2i), as in CODs


class PickEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg, split="train"):
        c = cfg["camera"]
        self.model = mujoco.MjModel.from_xml_string(build_scene_xml(cfg, with_can=True))
        self.data = mujoco.MjData(self.model)
        self.dropper = CanDropper(self.model, self.data, cfg)
        self.checker = SuctionChecker(self.model, cfg)
        self.cam = Camera(self.model, c["name"], c["width"], c["height_px"])
        self.objects = {self.model.geom("can").id}
        self.body = self.model.body("can").id
        self.seed_range = SPLITS[split]
        self.r_ok, self.r_fail = cfg["rl"]["reward_success"], cfg["rl"]["reward_fail"]
        self.observation_space = gym.spaces.Dict({
            "depth": gym.spaces.Box(0.0, np.inf, (c["height_px"], c["width"]), np.float32),
            "mask": gym.spaces.Box(0, 1, (GRID, GRID), np.uint8)})
        self.action_space = gym.spaces.Discrete(GRID * GRID)

    def _obs(self):
        return {"depth": self.depth, "mask": self.mask}

    def reset(self, *, seed=None, options=None):
        """seed: seeds the RNG that draws scene seeds. options={'scene_seed': s}: exact scene."""
        super().reset(seed=seed)
        scene = (options or {}).get("scene_seed")
        if scene is None:
            scene = int(self.np_random.integers(*self.seed_range))
        _, info = self.dropper.reset(scene)
        self.depth = self.cam.depth(self.data).astype(np.float32)
        self.K, T = self.cam.intrinsics(), self.cam.cam_to_world(self.data)
        self.cam_pos = T[:3, 3].copy()
        self.P = backproject(self.depth, self.K, T)
        self.mask = self.checker.valid_mask(self.P)[::2, ::2].astype(np.uint8)   # same rule as check()
        az = self.data.xmat[self.body].reshape(3, 3)[2, 2]
        pose = "upright" if az > 0.9 else "inverted" if az < -0.9 else "side" if abs(az) < 0.1 else "leaning"
        return self._obs(), {"scene_seed": scene, "pose": pose, "settle_steps": info["steps"]}

    def step(self, action):
        i, j = divmod(int(action), GRID)
        r = self.checker.check(self.data, self.P, self.depth, 2 * j, 2 * i, self.K, self.cam_pos, self.objects)
        reward = self.r_ok if r.ok else self.r_fail
        return self._obs(), reward, True, False, {"reason": r.reason, "cell": (i, j)}   # one-step episode

    def close(self):
        self.cam.close()