import sys
from functools import reduce
from multiprocessing import Pool


from pysc2 import maps
from pysc2.env import available_actions_printer
import run_loop
#from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import stopwatch


from absl import app
from absl import flags

from agent.random_agent import RandomAgent
# All model agents
from agent.model_agent_protocal import ModelAgent
from agent.atari_agent import AtariAgent

all_agent_classes = ["RandomAgent", "AtariAgent"]

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 64,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

flags.DEFINE_float("discount", 0.9, "Discount for reinforcement learning rate")
flags.DEFINE_integer("max_agent_steps", 2500, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent_name", "RandomAgent",
                    f"Which agent class to run, possible values: {all_agent_classes}")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(),
                  "Bot's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_bool("train", False, "Whether this is for training use.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

flags.DEFINE_string("map", None, "Name of a map to use.")

flags.DEFINE_string("model_dir", ".train-model", "Where to store model")
flags.DEFINE_string("summary_dir", ".train-summary", "Where to store summary")

flags.mark_flag_as_required("map")

def test_replay_buffer(rb):
    print('Episodes:',len(rb))
    total_transitions = 0
    for i, transition in enumerate(rb):
            print('Episode {i}: {transition} transitions'.format(i=i, transition=len(transition)))
            total_transitions += len(transition)
    print('Transitions in total', total_transitions)

def run_thread(agent_cls: ModelAgent.__class__, map_name, visualize):
    with sc2_env.SC2Env(
            map_name=map_name,
            agent_race=FLAGS.agent_race,
            bot_race=FLAGS.bot_race,
            difficulty=FLAGS.difficulty,
            step_mul=FLAGS.step_mul,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
            minimap_size_px=(FLAGS.minimap_resolution,
                             FLAGS.minimap_resolution),
            visualize=visualize) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        agent = agent_cls()
        if FLAGS.train is True:
            rb = run_loop.run_loop([agent], env, FLAGS.max_agent_steps, training=True)
            # test_replay_buffer(rb)
            agent.train_model(rb, FLAGS.model_dir, FLAGS.discount, FLAGS.summary_dir)
        else:
            run_loop.run_loop([agent], env, FLAGS.max_agent_steps, False, FLAGS.model_dir)
        
        if FLAGS.save_replay:
            env.save_replay(agent_cls.__name__)

        return agent.rewards


def main(unused_argv):
    """Run an agent."""
    pool = Pool(processes=FLAGS.parallel)

    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    maps.get(FLAGS.map)  # Assert the map exists.

    agent_cls = getattr(sys.modules[__name__], FLAGS.agent_name)

    async_results = [pool.apply_async(run_thread, (
        agent_cls, FLAGS.map, False)) for which in range(FLAGS.parallel)]

    # Can do anything here

    return_vals = [r.get() for r in async_results]

    all_scores = reduce(list.__add__, return_vals)
    print_stastic(all_scores)

    # After all threads done

    if FLAGS.profile:
        print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


def print_stastic(scorearray):
    avgscore = sum(scorearray)/len(scorearray)
    maxscore = max(scorearray)
    minscore = min(scorearray)
    print('Scores:', scorearray, file=sys.stderr)
    print('TotalEpisode:{episode}, AverageScore:{avg}, MaxScore:{max}'.format(
        episode=len(scorearray), avg=round(avgscore,2), max=maxscore), file=sys.stderr)
    return {'Eposide_num': len(scorearray), 'Avgscore': avgscore, 'Maxscore': maxscore, 'Minscore': minscore}


if __name__ == "__main__":
    app.run(main)
