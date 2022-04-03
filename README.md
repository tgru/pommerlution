# Pommerlution

A place to experiment with various reinforcement learning approaches in the [Pommerman](https://github.com/MultiAgentLearning/playground) environment.

The main goals of this project are leveraging self-play for training, evolutionary model selection of the best agents and exploring the usability of object embeddings for the game state.


## Prerequisites

- Python 3.9


## Development Setup
First clone this repository to a directory of your choice and move into it:

```bash
git clone https://github.com/tgru/pommerlution
cd pommerlution
```

Then it can be installed with development dependencies enabled and in editable mode, so code changes will take effect immediately:

```bash
pip install -e .[dev]
```


## Usage

Currently only checkpointless training of a single DQN agent in the `PommeFFACompetition-v0` environment is supported. All scripts can be found in the `bin` folder, while the library code resides in the `pommerlution` folder.

Training is done with the `train.py` script. Hyperparameters can be set there with the respective arguments. For example:

```bash
python ./bin/train.py --episodes 1000 --opponent simple_agent
```

 To get a list of all possible parameters and their description use the `--help` switch:

```bash
python ./bin/train.py --help
```