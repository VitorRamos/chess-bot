import gym
import numpy as np
import tensorflow as tf
import chess
import chess.pgn
import chess.svg

from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from stockfish import Stockfish
import chess.engine

from mcts import MCTS

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish")
stockfish = Stockfish("stockfish/stockfish", 
                        parameters={"Threads": 16,
                                    "Minimum Thinking Time": 20,
                                    "Hash": 256}, depth=15)

input = keras.Input(shape=(12, 8, 8), name='board')
x = layers.Conv2D(128, 5, padding='same', activation='relu')(input)
x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
tmp = layers.Conv2D(73, 1, padding='valid', activation='relu')(x)
p = layers.Dense(8*8*73, activation='softmax', name='p')(keras.layers.Flatten()(tmp))
v = layers.Dense(1, activation='tanh', name='v')(keras.layers.Flatten()(x))
model = keras.Model(
    inputs=[input],
    outputs=[p, v],
)
model.compile(
    optimizer='adam',
    loss=[
        keras.losses.CategoricalCrossentropy(),
        keras.losses.MeanSquaredError(),
    ],
)

# stockfish settings #
model_file = "./models/modelStockfish2000EloGames.h5"
accuracy_file = "./accuracy/move_accuracy_2000elo.txt"
stockfish_elo = 2000
####################

mfile = Path(model_file)
if mfile.is_file():
    model = keras.models.load_model(model_file)

afile = Path(accuracy_file)
if not mfile.is_file():
    open(accuracy_file, 'w').close()

env = gym.make('ChessAlphaZero-v0')

def getResult(str):
    if str == "0-1":
        return -1
    elif str == "1-0":
        return 1
    else:
        return 0

def generate_policy(val):
    policy = np.zeros(8 * 8 * 73)
    policy[val] = 1.0
    return policy

player = MCTS(model)
board = chess.Board()
turn = 1
correct_moves = 0
scores = []

training_positions = []
training_policies = []
training_results = []

while True:
    print("-------------------------------------")
    print("TURN: ", turn)
    white_move, policy = player.mcts(board, 50, True)
    training_positions.append(player.boardToBitBoard(board))
    training_policies.append(policy)

    stockfish.set_fen_position(board.fen())
    stockfish.set_elo_rating(1500)
    best_move1500 = stockfish.get_best_move()
    best_move1500 = chess.Move.from_uci(best_move1500)

    stockfish.set_elo_rating(2000)
    best_move2000 = stockfish.get_best_move()
    best_move2000 = chess.Move.from_uci(best_move2000)

    if(best_move1500 == white_move or best_move2000 == white_move):
        correct_moves += 1

    score_before = stockfish.get_evaluation()
    if score_before["type"] == "mate":
        score_before = 999*score_before["value"]
    else:
        score_before = score_before["value"]
    board.push(white_move)
    score_after = stockfish.get_evaluation()
    if score_after["type"] == "mate":
        score_after = 999*score_after["value"]
    else:
        score_after = score_after["value"]

    scores.append(score_after-score_before)

    print('white move ', white_move)
    print("White:")
    print(board)

    if not board.is_game_over():
        stockfish.set_fen_position(board.fen())
        stockfish.set_elo_rating(stockfish_elo)
        black_move = stockfish.get_best_move()
        black_move = chess.Move.from_uci(black_move)
        try:
            black_move_a = player.env.encode(black_move)
            training_positions.append(player.boardToBitBoard(board))
            training_policies.append(generate_policy(black_move_a))
        except:
            print('move not decoding: ', black_move)

        print('black move ', black_move)
        player.play_move(board, black_move)
        board.push(black_move)

        print("Black:")
        print(board)
        turn += 1

    if board.is_game_over():
        print(board.result())
        # calculate accurate move %
        move_percentage = round((correct_moves/turn) * 100, 2)
        print("Correct move %: ", move_percentage)
        # calculate move +/-
        avg = int(np.mean(scores))
        # write move accuracy to file
        f = open(accuracy_file, 'r+')
        lines = f.readlines()
        matches = len(lines)
        # format is: match:turns:w/l:move+/-:moveaccuracy
        f.write('\n'+str(matches)+':'+str(turn)+':'+str(board.result())+':'+str(avg)+':'+str(move_percentage))
        f.close()

        training_results = [ getResult(board.result()) for pos in training_positions ]

        training_positions = np.asarray(training_positions)
        training_policies = np.asarray(training_policies)
        training_results = np.asarray(training_results)

        model.fit(x=training_positions, y={"p": training_policies, "v": training_results}, epochs=7)
        model.save(model_file)

        player = MCTS(model)
        board = chess.Board()
        turn = 0
        correct_moves = 0
        scores = []
        training_positions = []
        training_policies = []