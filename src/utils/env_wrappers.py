import gymnasium as gym
from collections import deque
import numpy as np

# https://alexandervandekleut.github.io/gym-wrappers/
#env = gym.make('BipedalWalker-v3')

class BoxToHistoryBox(gym.ObservationWrapper):
    '''
    This wrapper converts the environment which returns last h observations.
    First h observations are converted such that first states are same.
    '''
    def __init__(self, env, h=8):
        super().__init__(env)
        self.h = h
        self.obs_memory = deque(maxlen=self.h)
        shape = (h,) + self.observation_space.shape
        low = np.repeat(np.expand_dims(self.observation_space.low, 0), h, axis=0)
        high = np.repeat(np.expand_dims(self.observation_space.high, 0), h, axis=0)    
        self.observation_space = gym.spaces.Box(low, high, shape)

    def add_to_memory(self, obs):
        # 处理新版本Gym返回的元组格式
        if isinstance(obs, tuple):
            obs = obs[0]  # 只取observation部分
        self.obs_memory.append(np.expand_dims(obs, axis=0))

    def observation(self, obs):
        # 处理新版本Gym返回的元组格式
        if isinstance(obs, tuple):
            obs = obs[0]  # 只取observation部分
        self.add_to_memory(obs)
        return np.concatenate(self.obs_memory)

    def reset(self):
        reset_state = self.env.reset()
        # 处理新版本Gym返回的元组格式
        if isinstance(reset_state, tuple):
            obs, info = reset_state
        else:
            obs, info = reset_state, {}
        for i in range(self.h-1):
            self.add_to_memory(obs)
        return self.observation(obs)

class MyWalkerWrapper(gym.Wrapper):
    '''
    This is custom wrapper for BipedalWalker-v3 and BipedalWalkerHardcore-v3. 
    Rewards for failure is decreased to make agent brave for exploration and 
    time frequency of dynamic is lowered by skipping two frames.
    '''
    def __init__(self, env, skip=2):
        super().__init__(env)
        self._obs_buffer = deque(maxlen=skip)
        self._skip = skip
        self._max_episode_steps = 750
        # 检测是否是gymnasium环境（通过检查step方法的返回值数量）
        try:
            test_action = env.action_space.sample()
            reset_result = env.reset()
            step_result = env.step(test_action)
            self._is_gymnasium = len(step_result) == 5
        except:
            self._is_gymnasium = False
        
    def step(self, action):
        total_reward = 0
        terminated = False
        truncated = False

        for i in range(self._skip):
            step_result = self.env.step(action)
            # 处理新版本gym的5个返回值 (obs, reward, terminated, truncated, info)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # 旧版本gym的4个返回值 (obs, reward, done, info)
                obs, reward, done, info = step_result
                terminated = done
                truncated = False

            if self.env.game_over:
                reward = -10.0
                info["dead"] = True
            else:
                info["dead"] = False

            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        # 根据环境类型返回正确格式的数据
        if self._is_gymnasium:
            return obs, total_reward, terminated, truncated, info
        else:
            return obs, total_reward, done, info

    def reset(self):
        reset_result = self.env.reset()
        # 处理新版本gym的返回格式 (obs, info)
        if isinstance(reset_result, tuple):
            return reset_result[0]  # 只返回observation部分
        else:
            return reset_result  # 旧版本直接返回observation

    def render(self, mode="human"):
        try:
            # 尝试新版本的render方法（不支持mode参数）
            for _ in range(self._skip):
                out = self.env.render()
        except TypeError:
            try:
                # 尝试旧版本的render方法（支持mode参数）
                for _ in range(self._skip):
                    out = self.env.render(mode=mode)
            except:
                # 如果都不行，返回None
                out = None
        return out
