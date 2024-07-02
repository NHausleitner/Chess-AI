import numpy as np
import chess
import chess.svg
from keras.src.saving import load_model

board = chess.Board()

colours = [chess.WHITE, chess.BLACK]
pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


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


model = load_model("ai_version1.keras")


def evaluate_board(board):
    return model.predict(np.array([fen_to_vec(board.fen())]))[0][0]


def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    legal_moves = list(board.legal_moves)

    if maximizing_player:
        max_eval = -np.inf
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = np.inf
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def find_best_move(board, depth):
    best_move = None
    best_value = -np.inf if board.turn == chess.WHITE else np.inf
    alpha = -np.inf
    beta = np.inf

    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        board.push(move)
        board_value = minimax(board, depth - 1, alpha, beta, not board.turn)
        board.pop()

        if board.turn == chess.WHITE:
            if board_value > best_value:
                best_value = board_value
                best_move = move
            alpha = max(alpha, board_value)
        else:
            if board_value < best_value:
                best_value = board_value
                best_move = move
            beta = min(beta, board_value)

        if beta <= alpha:
            break

    return best_move

implaying = 0
depth = 2  # You can adjust the search depth

while True:
    if implaying == 0:
        print(board)
        move = input("Your move: ")
        board.push_san(move)
        print(board)
        best_move = find_best_move(board, depth)
        print(f"AI move: {best_move}")
        board.push(best_move)
    else:
        print(board)
        best_move = find_best_move(board, depth)
        print(f"AI move: {best_move}")
        board.push(best_move)
        print(board)
        move = input("Your move: ")
        board.push_san(move)
