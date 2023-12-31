from flask import Flask, jsonify, request
from flask_cors import CORS
from agents.mcts import MCTS
from agents.ppo import PPO
import threading
import os

from environments.game import Game

games = {}
game_counter = 0
computer_agent_state = {}
computer_agent_model = {}

print(os.getcwd())

app = Flask(__name__)

cors = CORS(app, resources={r"*": {"origins": "*"}})

# this is an example of a route in flask
@app.route("/")
def hello_world():
    return "<h1>Hello, World!</h1>"

@app.route("/new_game")
def new_game():
    '''
    This route starts a new game.
    '''
    global game_counter
    global games
    global computer_agent_state
    global computer_agent_model
    game_id = game_counter
    game = Game()
    games[game_id] = game
    computer_agent_state[game_id] = 0
    helper = MCTS(num_iterations=1000)
    computer_agent_model[game_id] = PPO(name="ppo_v1", path="../data/models/ppo", helper=helper)
    game_counter += 1
    
    return jsonify({"game_state": game.make_json(), "game_id": game_id}), 200


@app.route("/get_game_state", methods=['GET'])
def get_game_state():
    '''
    Gets the current game state.
    '''
    global games
    global computer_agent_state
    id = request.args.get('id')
    game = games[int(id)]
    if computer_agent_state[int(id)] == 1:
        return jsonify({"game_state": game.make_json(), "agent_is_busy": True}), 200
    else:
        return jsonify({"game_state": game.make_json(), "agent_is_busy": False}), 200


@app.route("/play", methods=['POST'])
def play_game():
    '''
    To play a move.
    '''
    global games
    # Parse JSON data
    data = request.get_json()  
    # get game id from request body
    game_id = int(data['game_id'])
    move_game_idx = int(data['game_idx'])
    move_field_idx = int(data['field_idx'])
    move = (move_game_idx, move_field_idx)

    current_game = games[game_id]
    current_game.play(*move)
    # use threading to enable parallel / simultaneous games
    x = threading.Thread(target=move_agent_T, kwargs={'game_id': game_id})
    x.start()

    response = {
        "game_id": game_id,
        "game_state": current_game.make_json()
    }

    return response, 200


@app.route("/get_agent_state", methods=['GET'])
def get_agent_state():
    '''
    See if agent is busy making a move or not.
    '''
    global computer_agent_state
    id = request.args.get('id')
    return jsonify({"agent_is_busy": bool(computer_agent_state[int(id)])}), 200


def move_agent_T(game_id):
    '''
    Agent plays a move.
    T stands for Thread.
    '''
    global games
    global computer_agent_state
    global computer_agent_model
    game = games[game_id]
    computer_agent = computer_agent_model[game_id]
    computer_agent_state[game_id] = 1
    next_move = computer_agent.play(game)
    print(next_move)
    game.play(*next_move)
    computer_agent_state[game_id] = 0
    return True