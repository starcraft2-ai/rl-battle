# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified for 
"""A run loop for agent/environment interaction."""
import time


def run_loop(agents, env, max_frames=0, auto_call = 50, every_auto_call = None):
  """A run loop to have agents and an environment interact."""
  total_frames = 0
  start_time = time.time()

  action_spec = env.action_spec()
  observation_spec = env.observation_spec()
  for agent in agents:
    agent.setup(observation_spec, action_spec)

  try:
    while True:
      timesteps = env.reset()
      for a in agents:
        a.reset()
      while True:
        total_frames += 1
        actions = [agent.step(timestep)
                   for agent, timestep in zip(agents, timesteps)]
        if max_frames and total_frames >= max_frames:
          return
        if timesteps[0].last():
          break
        old_timesteps = timesteps
        timesteps = env.step(actions)
        [agent.after_step(old_timestep, timestep)
                   for agent, old_timestep, timestep in zip(agents, old_timesteps, timesteps)]
        if total_frames % auto_save is 0 and every_auto_call is not None:
          every_auto_call()
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f fps" % (
        elapsed_time, total_frames, total_frames / elapsed_time))
