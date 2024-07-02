import gc
import pickle
import sys
import random
import numpy as np
import chess
from keras import Sequential, Input
from keras.src.layers import Dense, BatchNormalization
from keras.src.saving import load_model
from stockfish import Stockfish

stockfish_path = "/Users/nickhausleitner/Documents/stockfish/stockfish-macos-m1-apple-silicon"
stockfish = Stockfish(stockfish_path)
stockfish.set_depth(12)
stockfish.update_engine_parameters({"Threads": 1})
print(stockfish.get_parameters())
colours = [chess.WHITE, chess.BLACK]
pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
gc.enable()


def fen_to_vec(fen):
    pos_fen, to_move, castling, en_passant, halfmove, fullmove = fen.split()
    board = chess.BaseBoard(pos_fen)
    piece_vectors = [np.array([1 if i in board.pieces(piece, colour) else 0 for i in range(64)], dtype=int)
                     for colour in colours for piece in pieces]
    board_vector = np.concatenate(piece_vectors)
    to_move_vector = np.array([to_move == 'w'], dtype=int)
    castling_vector = np.array([char in castling for char in 'KQkq'], dtype=int)
    en_passant_vector = np.zeros(64, dtype=int)
    if en_passant != '-':
        file = ord(en_passant[0]) - ord('a')
        rank = int(en_passant[1]) - 1
        en_passant_index = rank * 8 + file
        en_passant_vector[en_passant_index] = 1
    input_vector = np.concatenate([board_vector, to_move_vector, castling_vector, en_passant_vector])
    return input_vector


unique_fens = set()

"""
# Bei Bedarf auskommentieren
with open('unique_fens.pkl', 'rb') as f:
    unique_fens = pickle.load(f)
"""


def create_model1(shape):
    model = Sequential([
        Input(shape=(shape,)),
        Dense(2048, activation='relu'),
        BatchNormalization(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss='mse', metrics=['mae'])
    return model


def create_model2(shape):
    model = Sequential([
        Input(shape=(shape,)),
        Dense(837, activation='relu'),
        Dense(600, activation='relu'),
        Dense(500, activation='relu'),
        Dense(400, activation='relu'),
        Dense(300, activation='relu'),
        Dense(200, activation='relu'),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss='mse', metrics=['mae'])
    return model


board = chess.Board()


def generate_random_fen(games):
    board = chess.Board()
    fens = []
    for i in range(games):
        while not board.is_game_over():
            move = np.random.choice(list(board.legal_moves))
            board.push(move)
            fen = board.fen()
            if fen not in unique_fens:
                fens.append(fen)
                unique_fens.add(fen)
        board.reset()
    return fens


def generate_random_input(games):
    fens = generate_random_fen(games)
    vectors = []
    evaluations = []
    for fen in fens:
        stockfish.set_fen_position(fen)
        evaluation = stockfish.get_evaluation()
        if evaluation['type'] == 'cp':
            vec = fen_to_vec(fen)
            vectors.append(vec)
            evaluations.append(evaluation['value'])
    vectors = np.array(vectors)
    evaluations = np.array(evaluations)
    return vectors, evaluations


def train(games, epochs):
    for epoch in range(epochs):
        X_train, y_train = generate_random_input(games)

        print("Epoche" + str(epoch + 1))
        print("Das Set beinhaltet so viele FENs:f" + str(len(unique_fens)))
        print("Das Set verbraucht: " + str(sys.getsizeof(unique_fens) / (1024 * 1024)))

        model.fit(X_train, y_train, epochs=1, batch_size=8)
        print()
        print()

        if (epoch + 1) % 5 == 0:
            model.save("karl_version3.keras")
            print("karl gespeichert!")
            with open('unique_fens.pkl', 'wb') as f:
                pickle.dump(unique_fens, f)
            print("Set gespeichert!")

        print()
        gc.collect()


#model = load_model("karl_version1.keras")
model = create_model2(837)
games = 10
epochs = 20000
train(games, epochs)
