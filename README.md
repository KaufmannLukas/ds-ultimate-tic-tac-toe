# XOXO²

![xoxo](/images/slide_01_xoxo².png)  

## Introduction

The project was created during the final project phase of the Data Science Bootcamp at the Spiced Academy in Berlin in November 2023. <br>
The project goal was to use Reinforced Learning to teach an agent how to play Ultimate Tic-Tac-Toe (U_T-T-T). <br>
In this group project, we first created a Monte Carlo Tree Search (MCTS) search algorithm from scratch. Then we used an Artificial Neural Network (ANN) to implement Proximal Policy Optimization (PPO) to improve the performance of our agent. <br>
Addtionally, we implemented an interactive interface with flask and html, as a final product, where the user can play Ultimate Tic-Tac-Toe against our engine. <br>

## Rules of Ultimate Tic-Tac-Toe

![rules](/images/slide_02_rules_small.png)

Ultimate Tic-Tac-Toe (U_T-T-T) is played on nine tic-tac-toe boards arranged in a 3 × 3 grid. <br>
Playing on a field inside a board (local game), determines the next board in which the opponent must play their next move. <br>
The goal is to win three boards (local games) in a row. <br>
You can play your next move at any board, if you are directed to play in a full board or a board that has been won and therefore is already closed. <br>

![draw](/images/slide_03_draw_small.png)  

## Interface Example

![gif](/images/interface.gif) 

## About Reinforced Learning

Reinforced Learning is used not only in gaming environments, but also has use cases in robotics, self-driving cars and the development of generative AI.
We were interested in exploring this topic, since it was not part of the curriculum of our bootcamp and gained more and more importance over the years.

## Model types

![mcts+ppo](/images/slide_04_mcts+ppo.png)
As mentioned before, we first created a Monte Carlo Tree Search Algorithm (MCTS) from scratch and additionally implemented a memory file from former iterations that the model can load and therefore learn from. <br>
As a second model, we created a Neural Network structure to perform Proximal Policy Optimization (PPO).

## MCTS

![mcts](/images/slide_05_mcts.png)

The MCTS works as follows: One iteration is completed after selecting a move, expanding and exploring the game tree (all nodes in the tree are possible game states or moves that can be played) and simulating the outcome from there by random play. The outcome of this playout is than backpropagated to the top, along the path that has been played. The visit counts and reward values of these nodes are then updated. In order to access these results in the future, we implemented a memory file. This file grows large in size quickly, though.

## PPO

![ppo](/images/slide_06_ppo.png)

In the PPO model, our agent performs an action (=move) in the environment (=game) and gets a new game state, as well as rewards for his actions. The actor network predicts probabilites for each move and the critic network estimates the value of the given board state.

## Model evaluation

![round1](/images/slide_07_round1.png)

In order to test the performance of our agents, we let them play U_T-T-T against our baseline model, which performs random moves, as well as against each other.

![results1](/images/slide_08_results1.png)

Finally, we put our two strongest agents - MCTS with memory and PPO+MCTS - in the ring to play against each other.

![round2](/images/slide_09_round2.png)

The final results show, that our PPO+MCTS agent has the strongest performance and wins about 60% of the games against MCTS with memory.

![results2](/images/slide_10_results2.png)

It is therefore our current strongest agent and connected with the interface.

## Files and Folders

The following tree-like folder structure diagram provides orientation over our repository:

![folders](/images/slide_11_folders.png)

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


## Docker file

...