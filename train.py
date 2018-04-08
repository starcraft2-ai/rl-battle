import Environment
import Models


import argparse

parser = argparse.ArgumentParser(description='Multi-Agent Reinforcement learning on StarCraft 2 Trainer')
parser.add_argument('--episodes', '-e', type=int, default=100,
                    help='Total episode to reinforce')
parser.add_argument('--max-steps', '-s', type=int, default=800,
                    help='maximium steps to run on game')
parser.add_argument('--map', '-m', default='',
                    help='map to run the game')

args = parser.parse_args()
