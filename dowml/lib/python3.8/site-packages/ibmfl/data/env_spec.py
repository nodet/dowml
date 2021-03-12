"""
IBM Confidential
OCO Source Materials
5737-H76, 5725-W78, 5900-A1R
(c) Copyright IBM Corp. 2020 All Rights Reserved.
The source code for this program is not published or otherwise divested of its trade secrets,
irrespective of what has been deposited with the U.S. Copyright Office.
"""
"""
EnvHandler for OpenAI gym interface
"""
import abc

import gym


class EnvHandler(gym.Env):
    """
    Base class for Environment Handler of Reinforcement learning algorithms
    """

    def __init__(self, data=None, env_config=None):
        """
        Initializes an `EnvHandler` object

        :param config: Start state configuration of environment
        :type config: `dict`
        """

    @abc.abstractmethod
    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further
             step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information
            (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
       Returns:
           observation (object): the initial observation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, mode='human'):
        """
        Render one frame of the environment
        """
        raise NotImplementedError
