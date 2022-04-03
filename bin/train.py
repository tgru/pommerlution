#!/usr/bin/env python
import argparse
import datetime
import logging
import random
import sys

import pommerman
import tensorflow as tf

from pommerlution.agents.agent_factory import AgentFactory
from pommerlution.models.dense import MLP
from pommerlution.models.dqn import DQN
from pommerlution.training import ExperienceMemory, create_replays


def train(config: argparse.Namespace):
    """
    Train a model based on the given configuration.

    :param config: Configuration values for training
    """
    min_capacity = config.minMemorySize
    max_capacity = config.maxMemorySize
    memory = ExperienceMemory(capacity=max_capacity)
    environment = "PommeFFACompetition-v0"

    net_config = (
        [121, 64, 32, 16, 8, 6],
        ["relu", "relu", "relu", "relu", "relu", "linear"],
    )
    target_net = MLP(*net_config)
    train_net = MLP(*net_config)
    train_net.build((None, 121))

    optimizer = tf.optimizers.Adam(
        learning_rate=config.learningRate
    )

    dqn = DQN(train_net, target_net, optimizer, epsilon=0.1)
    dqn.sync()

    agent_factory = AgentFactory()
    agent = agent_factory.get_agent("dqn_agent")
    agent.model = dqn

    # Create four opponents and replace a random one with own agent
    agents = [agent_factory.get_agent(config.opponent) for _ in range(4)]
    agent_idx = random.randint(0, 3)
    agents[agent_idx] = agent

    env = pommerman.make(environment, agents)

    for episode in range(config.episodes):
        # Create new experiences
        total_reward = 0
        num_steps = 0

        if len(memory) < min_capacity:
            logging.info(f"Gathering {min_capacity} experiences before starting training")

        while True:
            replays = create_replays(env, agents)
            replay = replays[agent_idx]
            for experience in replay.experiences:
                memory.add(experience)
            total_reward = replay.total_reward()
            num_steps = len(replay.experiences)

            # Wait until minimum capacity is reached before training
            if len(memory) >= min_capacity:
                break

        # Train the model
        batch = memory.sample(config.batchSize)
        loss = agent.model.train(batch, epochs=1)

        logging.info(
            f"Episode {episode+1}/{config.episodes}; Avg. Reward: {total_reward/num_steps:0.5f}; Steps: {num_steps:3d}; Loss: {loss:0.5f}")

        if not (episode+1) % config.syncInterval:
            logging.info("Synchronizing target network")
            agent.model.sync()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"DQN training script")

    parser.add_argument("--runName", dest="runName", metavar="", type=str,
                        help="Name of the run for logging purposes. Defaults to current timestamp.",
                        default=datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S"))

    levels = [name for name in logging._nameToLevel]
    parser.add_argument("--logLevel",  dest="logLevel", metavar="", type=str,
                        help="Log level to use. One of [%(choices)s]",
                        choices=levels,
                        default="INFO")

    parser.add_argument("--opponent",  dest="opponent", metavar="", type=str,
                        help=f"Opponent type to train against. One of [%(choices)s]",
                        choices=["random_agent", "simple_agent"],
                        default="random_agent")

    parser.add_argument("--episodes",  dest="episodes", metavar="", type=int,
                        help=f"Number of episodes to train.",
                        default=100)

    parser.add_argument("--batchSize",  dest="batchSize", metavar="", type=int,
                        help=f"Size of batches used for training",
                        default=32)

    parser.add_argument("--learningRate",  dest="learningRate", metavar="", type=float,
                        help=f"Fixed learning rate used by the optimizer",
                        default=1e-3)

    parser.add_argument("--minMemorySize",  dest="minMemorySize", metavar="", type=int,
                        help=f"Minimum number of experiences in the memory before training starts",
                        default=2**10)

    parser.add_argument("--maxMemorySize",  dest="maxMemorySize", metavar="", type=int,
                        help=f"Maximum number of experiences in the memory",
                        default=2**32)

    parser.add_argument("--syncInterval",  dest="syncInterval", metavar="", type=int,
                        help=f"Interval between target network synchronizations",
                        default=20)

    config = parser.parse_args(sys.argv[1:])

    logging.basicConfig(
        level=logging.getLevelName(config.logLevel),
        format="%(asctime)s.%(msecs)03d: %(levelname).1s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logging.info(f"Starting training run {config.runName}")
    train(config)
    logging.info("Finished training!")
