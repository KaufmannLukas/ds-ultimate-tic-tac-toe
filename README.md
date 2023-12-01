# ds-ultimate-tic-tac-toe

  - parts of presentation (intro)
  - files and folders
  - gif gameplay interface
  - setup environment
  - setup interface

## Introduction

The project ... (description)

## About Reinforced Learning

... information

## Files and Folders

... orientation

.
├── data/
│   ├── models/
│   │   ├── mcts/
│   │   │   ├── mcts_ltmm_02.pkl
│   │   │   └── ...
│   │   └── ppo/
│   │       ├── ppo_v_ppo_v1_7_actor.pth
│   │       ├── ppo_v_ppo_v1_7_critic.pth
│   │       └── ...
│   └── ...
├── EDA.ipynb
├── frontend/
│   └── ds-uttt (*necessary?)/
│       └── ...
├── README.md
├── src/
│   ├── agents/
│   │   ├── agent.py
│   │   ├── human.py
│   │   ├── random.py
│   │   ├── mcts.py
│   │   ├── ppo.py
│   │   └── network/
│   │       └── network.py
│   ├── environments/
│   │   ├── game
│   │   └── uttt_env
│   ├── tests/
│   │   ├── test_game.py
│   │   └── test_mcts.py
│   ├── main_flask.py
│   ├── mcts_train.py
│   ├── ppo_train.py
│   └── main_play.py
└── docker-file (*?)

## Setup Environment

To set up with anaconda / mamba:

``` bash
conda env create --file environment.yml
```

To update the environment:

``` bash
conda env update --file environment.yml
```

To activate the environment:

``` bash
conda activate ds-uttt
```

## Interface Example

...