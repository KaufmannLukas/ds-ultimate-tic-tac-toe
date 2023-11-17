from flask import Flask, jsonify, request
import os


from .environments.game import Game
games = {}
game_counter = 0

print(os.getcwd())

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<h1>Hello, World!</h1>"

@app.route("/new_game")
def new_game():
    global game_counter
    global games
    game_id = game_counter
    games[game_id] = Game()
    game_counter += 1
    return jsonify(game_id), 200


@app.route("/get_game_state", methods=['GET'])
def get_game_state():
    global games
    id = request.args.get('id')
    game = games[int(id)]
    return jsonify(game.blocked_fields.tolist())

@app.route("/play")
def play_game():
    games[game_counter] = Game()
    game_counter += 1
    return 