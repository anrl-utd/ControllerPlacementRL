"""
Sets up environment into openai-gym registry
"""
from setuptools import setup

setup(name='controller_env',
    version='0.0.1',
    install_requires=['gym', 'networkx', 'matplotlib', 'stable-baselines[mpi]', 'optuna']	#And any other dependencies required
)