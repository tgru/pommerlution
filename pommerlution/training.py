from collections import deque
from collections import namedtuple
from enum import Enum
import random

from gym.core import Env
from pommerman.agents import BaseAgent

Experience = namedtuple("Experience",
                        ("state", "action", "next_state", "reward", "done"))


class GameResult(Enum):
    """
    Constants for game results.
    """
    Lose = -1
    Draw = 0
    Win = 1


class Replay():
    """
    Summarizes an episode from the point of view of an agent.
    """

    def __init__(self) -> None:
        """
        Initializes an empty `Replay`.
        """
        self.experiences: list[Experience] = []
        self.end: GameResult = None

    def total_reward(self) -> float:
        """
        Calculate and return the summed reward of all saved experiences.

        :return: The sum of all rewards
        """
        return sum([experience.reward for experience in self.experiences])


class ExperienceMemory():
    """
    Container to safe state-action pairs.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.experiences: deque = deque([], maxlen=capacity)

    def add(self, experience: Experience) -> None:
        """
        Add an `Experience` to the memory.
        """
        self.experiences.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        """
        Return random experiences from the memory.

        :return: A list of experiences of size `batch_size` 
        """
        return random.sample(self.experiences, batch_size)

    def __len__(self) -> int:
        return len(self.experiences)


def create_replays(env: Env, agents: list[BaseAgent]) -> list[Replay]:
    """
    Run an episode with the given agents and return the gathered
    experiences of everyone as `Replay` objects.

    :param environment: An Pommerman environment
    :param agents: List of agents to use

    :return: A list containing a Replay per agent
    """
    replays = [Replay() for _ in range(len(agents))]
    done = False
    state = env.reset()

    while not done:
        actions = env.act(state)
        next_state, reward, done, _ = env.step(actions)
        if done:
            end = reward

        for i in range(len(agents)):
            replays[i].experiences.append(Experience(
                state[i], actions[i], next_state[i], reward[i], done))
            if done:
                replays[i].end = end
        state = next_state
    return replays
