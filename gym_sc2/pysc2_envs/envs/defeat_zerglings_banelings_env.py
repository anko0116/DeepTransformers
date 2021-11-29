from typing import AsyncIterable
import gym
from gym import spaces
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import logging
import numpy as np

# Credit: https://towardsdatascience.com/create-a-customized-gym-environment-for-star-craft-2-8558d301131f


logger = logging.getLogger(__name__)

class DZBEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    '''
    'agent_interface_format': features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=64),
    '''

    default_settings = {
    'map_name': "CollectMineralShards",
    'players': [sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.hard)],
    'agent_interface_format': sc2_env.parse_agent_interface_format(
        feature_screen=84,
        feature_minimap=64,
        rgb_screen=None,
        rgb_minimap=None,
        action_space=None,
        use_feature_units=False,
        use_raw_units=False),
    'realtime': False,
    #'visualize': True,
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.env = None
        # 0 no operation
        # 1~32 move
        # 33~122 attack
        self.action_space = spaces.Box(low=1.0, high=84.0, shape=(2,), dtype=np.float32) # Coordinates of Attack
        # [0: x, 1: y, 2: hp]
        self.image_shape = (1, 84, 84)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.image_shape
        )
        self.num_steps = 0
        self.available_actions = None

    def step(self, action, args=None):
        #raw_obs = self.take_action(actions.FunctionCall(actions.FUNCTIONS.select_army.id, [[0]])) # take safe action
        if len(self.available_actions) and actions.FUNCTIONS.Attack_screen.id in self.available_actions:
            raw_obs = self.env.step([actions.FunctionCall(actions.FUNCTIONS.Attack_screen.id, [[0],[action[0],action[1]]])])
        else:
            raw_obs = self.env.step([actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])])
        raw_obs = raw_obs[0]
        self.num_steps += 1
        self.available_actions = None if "available_actions" not in raw_obs.observation.keys() else raw_obs.observation["available_actions"]

        done = False
        if self.num_steps == 50:
            obs = self.reset()
            done = True

        return self.get_obs(raw_obs), raw_obs.reward, done, {}  # return obs, reward and whether episode ends

    def reset(self):
        if self.env is None:
            self.init_env()

        raw_obs = self.env.reset()[0] # 0-indexed because raw_obs is tuple for no reason
        self.available_actions = raw_obs.observation["available_actions"]
        self.num_steps = 0
        '''
        for action in raw_obs.observation.available_actions:
            print(actions.FUNCTIONS[action])
        print("@@@")
        '''
        return self.get_obs(raw_obs)

    def get_obs(self, raw_obs):
        obs = raw_obs.observation["feature_screen"][5]
        return np.reshape(obs, self.image_shape)

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()

    def render(self, mode='human', close=False):
        pass

    """ HELPER FUNCTIONS BELOW """

    def save_replay(self, replay_dir, prefix=None):
        replay_path = self.env.save_replay(replay_dir, prefix=prefix)
        return replay_path

    def init_env(self):
        """Used in self.reset()"""
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

    def get_derived_obs(self, raw_obs):
        """Used in self.reset() and self.step()"""
        obs = np.zeros((19,3), dtype=np.uint8)
        # 1 indicates my own unit, 4 indicates enemy's
        marines = self.get_units_by_type(raw_obs, units.Terran.Marine, 1)
        zerglings = self.get_units_by_type(raw_obs, units.Zerg.Zergling, 4)
        banelings = self.get_units_by_type(raw_obs, units.Zerg.Baneling, 4)
        self.marines = []
        self.banelings = []
        self.zerglings = []
        for i, m in enumerate(marines):
            self.marines.append(m)
            obs[i] = np.array([m.x, m.y, m[2]])
        for i, b in enumerate(banelings):
            self.banelings.append(b)
            obs[i+9] = np.array([b.x, b.y, b[2]])
        for i, z in enumerate(zerglings):
            self.zerglings.append(z)
            obs[i+13] = np.array([z.x, z.y, z[2]])
        return obs

    def take_action(self, action):
        """Used in self.step()"""
        # map value to action
        if action == 0:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
        elif action<=32:
            derived_action = np.floor((action-1)/8)
            # get unit idx
            idx = (action-1)%8
            if derived_action == 0:
                action_mapped = self.move_up(idx)
            elif derived_action == 1:
                action_mapped = self.move_down(idx)
            elif derived_action == 2:
                action_mapped = self.move_left(idx)
            else:
                action_mapped = self.move_right(idx)
        else:
            # get attacker and enemy unit idx
            eidx = np.floor((action-33)/9)
            aidx = (action-33)%9
            action_mapped = self.attack(aidx, eidx)
        # execute action
        raw_obs = self.env.step([action_mapped])
        return raw_obs

    def move_up(self, idx):
        idx = np.floor(idx)
        try:
            selected = self.marines[idx]
            new_pos = [selected.x, selected.y-2]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_down(self, idx):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x, selected.y+2]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_left(self, idx):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x-2, selected.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_right(self, idx):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x+2, selected.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def attack(self, aidx, eidx):
        try:
            selected = self.marines[aidx]
            if eidx>3:
                # attack zerglines
                target = self.zerglines[eidx-4]
            else:
                target = self.banelings[eidx]
            return actions.RAW_FUNCTIONS.Attack_unit("now", selected.tag, targeted.tag)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def get_units_by_type(self, obs, unit_type, player_relative=0):
        """
        NONE = 0
        SELF = 1
        ALLY = 2
        NEUTRAL = 3
        ENEMY = 4
        """
        return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == player_relative]