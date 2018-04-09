import Environment
from Models import ActorNetwork, CriticNetwork
from utils import *

import argparse

parser = argparse.ArgumentParser(description='Multi-Agent Reinforcement learning on StarCraft 2 Trainer')
parser.add_argument('--episodes', '-e', type=int, default=100,
                    help='Total episode to reinforce')
parser.add_argument('--maxsteps', '-s', type=int, default=800,
                    help='maximium steps to run on game')
parser.add_argument('--map', '-m', default='',
                    help='map to run the game')

args = parser.parse_args()

import tensorflow as tf

tf.enable_eager_execution()

def main():
    actor  = ActorNetwork()
    critic = CriticNetwork()

    target_actor  = ActorNetwork()
    target_critic = CriticNetwork()
    
    R = Buffer()

    for episodes in range(args.episodes):
        # TODO:

        for t in range(args.maxsteps):
            pass


    

if __name__ == '__main__':
    main()