import os
from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions
import tensorflow as tf
from tensorflow.contrib import eager as tfe
tfe.enable_eager_execution()
from Networks.atari import AtariModel


possible_action_num = len(actions.FUNCTIONS)


class AtariAgent(BaseAgent):
    def __init__(self, name='AtariAgent'):
        super().__init__()
        self.name = name
        self.model: AtariModel = None

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)

    def reset(self):
        super().reset()

    def step(self, obs):
        super().step(obs)
        (screen, minimap, info) = (
            tfe.Variable(obs.observation['screen']),
            tfe.Variable(obs.observation['minimap']),
            tfe.zeros([possible_action_num])
        )
        info[obs.observation['available_actions']] = 1

        # induce dimension
        x = (
            tf.expand_dims(minimap, 0),
            tf.expand_dims(screen, 0),
            tf.expand_dims(info, 0)
        )

        # predict
        (spatial_action, non_spatial_action, value) = self.model.predict(x)

        # reduce dimentsion
        y, x = spatial_action
        y, x = y[0], x[0]
        non_spatial_action = non_spatial_action[0]
        value = value[0]

        # select available_actions
        action_selected = tf.argmax(non_spatial_action * info[0])

        # form action and call
        # TODO: better implementation
        act_args = []
        for arg in actions.FUNCTIONS[action_selected].args:
            if arg.name in ('screen', 'minimap'):
                act_args.append([x, y])
            else:
                act_args.append([0])
        return actions.FunctionCall(action_selected, act_args)

    def build_model(self, initializer=tf.zeros):
        self.model = AtariModel(
            self.obs_spec["screen"][0], self.obs_spec["minimap"][0], possible_action_num)

        # TODO: Training
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.root = tfe.Checkpoint(optimizer=optimizer,
                                   model=self.model,
                                   optimizer_step=tf.train.get_or_create_global_step())

    def load_model(self, checkpoint_dir):
        self.root.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def save_model(self, checkpoint_dir):
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.root.save(file_prefix=checkpoint_prefix)
