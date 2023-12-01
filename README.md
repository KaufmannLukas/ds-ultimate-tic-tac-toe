# ds-ultimate-tic-tac-toe

TODO's:
  - parts of presentation (intro)
  - files and folders
  - gif gameplay interface
  - setup environment
  - setup interface

## Introduction

The project was created during the final project phase of the Data Science Bootcamp at the Spiced Academy in Berlin in November 2023. <br>
The project goal was to use Reinforced Learning to teach an agent how to play Ultimate Tic-Tac-Toe (U_T-T-T). <br>
In this group project, we first created a Monte Carlo Tree Search (MCTS) search algorithm from scratch. Then we used an Artificial Neural Network (ANN) to implement Proximal Policy Optimization (PPO) to improve the performance of our agent. <br>
Addtionally, we implemented an interactive interface with flask and html, as a final product, where the user can play Ultimate Tic-Tac-Toe against our engine. <br>

## Rules of Ultimate Tic-Tac-Toe

Ultimate Tic-Tac-Toe (U_T-T-T) is played on nine tic-tac-toe boards arranged in a 3 × 3 grid. <br>
Playing on a field inside a board (local game), determines the next board in which the opponent must play their next move. <br>
The goal is to win three boards (local games) in a row. <br>
You can play your next move at any board, if you are directed to play in a full board or a board that has been won and therefore is already closed. <br>

## About Reinforced Learning

Reinforced Learning is used not only in gaming environments, but also has use cases in robotics, self-driving cars and the development of generative AI.
We were interested in exploring this topic, since it was not part of the curriculum of our bootcamp and gained more and more importance over the years.

## Files and Folders

The following tree diagram provides orientation over our repository:

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
├── environment.yml
├── frontend/
│   └── ds-uttt (*necessary?)/
│       └── ...
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
├── README.md
└── docker-file (*?)
.

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

## Setup Interface

TO RUN GAME (backend):
1. use the command: "flask --debug --app main_flask run"
(Note: make sure you are in the correct folder)

2. open a new terminal window, go to ../frontend/ds-uttt

TO RUN GAME (frontend):
3. use the command: "npm run dev"

4. click or copy the URL: "http://localhost:5173/"

5. The game should be running in your browser.
(Note: if you experience some graphical issues, try another browser)


## Interface Example

...


## Docker file

...