import sys
from functools import reduce
from multiprocessing import Pool, Lock


from pysc2 import maps
from pysc2.env import available_actions_printer
import train_runloop
from pysc2.env import sc2_env
from pysc2.lib import stopwatch


from absl import app
from absl import flags

# All model agents
from agent.model_agent_protocal import ModelAgent
from agent.atari_agent import AtariAgent

from Environment import A2CEnvironment

all_agent_classes = ["AtariAgent"]

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 84,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 84,
                     "Resolution for minimap feature layers.")

flags.DEFINE_integer("game_steps_per_episode", 2500, "Game steps per episode.")
flags.DEFINE_integer("max_agent_steps", 2500 * 100, "Total agent steps.")

flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent_name", "RandomAgent",
                    f"Which agent class to run, possible values: {all_agent_classes}")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(),
                  "Bot's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

flags.DEFINE_string("map", None, "Name of a map to use.")
flags.mark_flag_as_required("map")

# Multi Process things
lock = Lock()
agent_model = None
(optimizer, root_node) = None, None
replay_buffer = []

def run_thread(agent_cls: ModelAgent.__class__, map_name, visualize):
    global lock, agent_model, replay_buffer, optimizer, root_node
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
        (action_spec, observation_spec) = (
            env.action_spec(),
            env.observation_spec()
            )

        agent_env = A2CEnvironment(lock, agent_cls)
        agent_env.set_sepcs(action_spec, observation_spec)
        
        agent_env.set_replay_buffer(replay_buffer)
        with lock:
            if agent_model is None:
                agent_model = agent_env.build_model()
                (optimizer, root_node) = agent_env.get_optimizer_and_node()
        agent_env.set_model(agent_model, optimizer, root_node)


        train_runloop.run_loop([agent_env], env, FLAGS.max_agent_steps)

        if FLAGS.save_replay:
            env.save_replay(agent_cls.__name__)

        return agent_env


def main(unused_argv):
    """Run an agent."""
    pool = Pool(processes=FLAGS.parallel, initargs=(lock, ))

    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    maps.get(FLAGS.map)  # Assert the map exists.

    agent_cls = getattr(sys.modules[__name__], FLAGS.agent_name)

    async_results = [pool.apply_async(run_thread, (
        agent_cls, FLAGS.map, False)) for which in range(FLAGS.parallel)]

    # Can do anything here

    agent_envs = [r.get() for r in async_results]

    last_scores = [env.agent.reward for env in agent_envs]

    print_stastic(last_scores)

    # After all threads done

    if FLAGS.profile:
        print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


def print_stastic(last_scores):
    avgscore = sum(last_scores)/len(last_scores)
    maxscore = max(last_scores)
    minscore = min(last_scores)
    print('Last Scores:', last_scores, file=sys.stderr)
    print(f"\
            In Parallel:{len(last_scores)}\n\
            AverageScore:{round(avgscore, 2)}\n\
            MaxScore:{maxscore}\n\
            ", file=sys.stderr)
    return {'Eposide_num': len(last_scores), 'Avgscore': avgscore, 'Maxscore': maxscore, 'Minscore': minscore}


if __name__ == "__main__":
    app.run(main)
