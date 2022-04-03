import random

import tensorflow as tf

from pommerlution.training import Experience


class DQN(tf.keras.Model):
    def __init__(self,
                 train_net: tf.keras.Model,
                 target_net: tf.keras.Model = None,
                 optimizer:  tf.keras.optimizers.Optimizer = None,
                 epsilon: float = 0.0,
                 gamma: float = 0.99,
                 double_dqn: bool = True,
                 ) -> None:
        """
        Initializes the DQN model for training or inference.

        :param train_net: Model to train or for inference
        :param target_net: Model to derive target values during training
        :param optimizer: Optimizer for training. Optional for inference
        :param epsilon: Probability of infering a random action
        :param gamma: Gamma value used for target calculation
        :param double_dqn: Train with double DQN if true
        """
        super(DQN, self).__init__()

        self.train_net = train_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.gamma = gamma
        self.double_dqn = double_dqn

    def sync(self) -> None:
        """
        Synchronize the target network with the training network.
        """
        self.target_net.set_weights(self.train_net.get_weights())

    def should_explore(self) -> bool:
        """
        Return whether the agent should choose a random action or not.

        :return: `True` if a random action should be chosen, else `False`
        """
        if random.random() < self.epsilon:
            return True
        else:
            return False

    def random_action(self) -> int:
        """
        Return a random action based on the given gym `action_space`.

        :return: A random action
        """
        return random.randint(0, 5)

    def policy(self, obs: dict) -> int:
        """
        Return the epsilon-greedy policy based action based on the
        observation.

        :param obs: Environment observation to act on

        :return: The policy based action
        """
        x = tf.constant(obs["board"].flatten())
        x = tf.expand_dims(x, axis=0)

        if self.should_explore():
            return self.random_action()
        else:
            return self.call(x)

    def call(self, x: tf.Tensor) -> int:
        """
        Return the model based action to execute based on `x`.

        :param x: Environment observation to act on

        :return: An integer defining the action
        """
        y = self.train_net(x)

        return int(tf.argmax(y, axis=1))

    def train(self, batch: list[Experience], epochs: int = 1) -> None:
        """
        Train the network on a batch of `Transition` tuples.

        :param batch: A list of `Transition` tuples to train on
        :param epochs: Number of gradient steps to apply
        """

        # Prepare inputs
        batch_size = len(batch)

        x_0 = [sample.state["board"].flatten() for sample in batch]
        x_0 = tf.constant(x_0)

        x_1 = [sample.next_state["board"].flatten() for sample in batch]
        x_1 = tf.constant(x_1)

        actions = [sample.action for sample in batch]

        rewards = [sample.reward for sample in batch]
        rewards = tf.constant(rewards, dtype=tf.float32)

        done = [float(sample.done) for sample in batch]
        done = tf.constant(done, dtype=tf.float32)

        if self.double_dqn:
            # Get the selected actions from training net
            indices = [[action] for action in actions]

            # Choose target Q-values based on the actions indices
            q_target = self.target_net(x_1)
            q_target = tf.gather_nd(q_target, indices, batch_dims=1)
        else:
            # Take the target value directly
            q_target = self.target_net(x_1)
            q_target = tf.reduce_max(q_target, axis=1)
        y_1 = rewards + self.gamma * q_target * (1-done)

        with tf.GradientTape() as tape:
            y_0 = tf.reduce_max(self.train_net(x_0), axis=1)
            loss = (y_1-y_0)**2  # MSE
            loss = tf.reduce_sum(loss)/batch_size  # Average loss over batch
        gradients = tape.gradient(loss, self.train_net.trainable_weights)
        for _ in range(epochs):
            self.optimizer.apply_gradients(
                zip(gradients, self.train_net.trainable_weights))

        return loss
