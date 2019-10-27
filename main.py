import gym
import controller_env

env = gym.make('Controller-v0')
env.step(0)
env.reset()
env.render()