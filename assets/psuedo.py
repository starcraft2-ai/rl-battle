def main():
    actor         = ActorNetwork(), critic        = CriticNetwork()
    target_actor  = ActorNetwork(), target_critic = CriticNetwork()
    
    R = Buffer()

    env = Environment().init_env()

    for episodes in 1..max_episodes:
        env.reset_env()

        state = env.get_current_state(t)
        action_queue = Queue()
        for t in 1..max_steps:
            if action_queue is not empty:
                if not env.step(action_queue.dequeue()):
                    # game over
                    break
            else:
                best_action = actor.forward(state)
                action_queue.enqueue(best_action)

                reward = env.get_reward()
                new_state = env.get_current_state()

                transition = (state, best_action, new_state)
                R.add(transition)
                M = R.sample(args.sample_batch_size)
                
                Q = critic.forward(state, best_action)
                Q^ = for transition in M:
                        trans_state_next = transition
                        trans_best_action_next = target_actor.forward(trans_state_next)
                        
                        target_critic.forward(trans_state_next, trans_best_action_next) * args.lambdaa + reward
                
                cost = MeanSquareError(Q^, Q)
                actor.back_propergate(reward), critic.back_propergate(cost)

                target_actor  = target_actor  * (1 - args.learning_rate) + args.learning_rate * actor
                target_critic = target_critic * (1 - args.learning_rate) + args.learning_rate * critic

                state = new_state