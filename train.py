from Environment import Environment
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
parser.add_argument('--screensize', type=int, default=84,
                    help='screen resolution')
parser.add_argument('--minisize', type=int, default=64,
                    help='minimap resolution')
parser.add_argument('--sample_batch_size', '-b', type=int, default=8,
                    help='sample_batch_size')
parser.add_argument('--lambdaa', type=float, default=0.2,
                    help='Discount rate') 
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='Learning rate') 
args = parser.parse_args()

import tensorflow as tf
tf.enable_eager_execution()

def main():
    actor  = ActorNetwork()
    critic = CriticNetwork()

    target_actor  = ActorNetwork()
    target_critic = CriticNetwork()
    
    R = Buffer()

    env = Environment(screen_size = args.screensize, minimap_size = args.minisize)
    env.init_env()

    for episodes in range(args.episodes):
        env.reset_env()
        state = env.get_current_state()
        (common_state, agents_state) = state

        for t in range(args.maxsteps):
            (actions_table, coodinates) = actor.forward(common_state, agents_state)
            best_action = best_actions(actions_table)
            env.transit(best_action)

            reward = env.get_reward()
            new_state = env.get_current_state()

            R.add((state, best_action, new_state))
            M = R.sample(args.sample_batch_size)

            Qs = []
            for transition in M:
                (trans_state, _, trans_state_next) = transition
                (trans_common_state_next, trans_agents_state_next) = trans_state_next
                (trans_actions_table_next, trans_coodinates_next) = target_actor.forward(trans_common_state_next, trans_agents_state_next)
                trans_best_action_next = best_actions(trans_actions_table_next)
                
                Qs.append(
                    target_critic.forward(
                        trans_common_state_next, 
                        trans_agents_state_next, 
                        trans_best_action_next) * args.lambdaa + reward
                )
            
            # TODO: BP

            target_actor = target_actor * (1 - args.learning_rate) + args.learning_rate * actor
            target_critic = target_critic * (1 - args.learning_rate) + args.learning_rate * critic

if __name__ == '__main__':
    main()