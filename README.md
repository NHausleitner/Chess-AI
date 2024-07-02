# Chess Evaluation Model Training

This repository contains code for training a chess position evaluation model using Stockfish and a neural network built with Keras. The model aims to evaluate chess positions given in FEN (Forsyth-Edwards Notation) format by predicting evaluation scores similar to those provided by Stockfish.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Functions and Classes](#functions-and-classes)
- [Training the Model](#training-the-model)
- [Saving and Loading Models](#saving-and-loading-models)

## Installation

To run this project, ensure you have the following dependencies installed:

- Python 3.8+
- numpy
- keras
- python-chess
- stockfish

You can install the required Python packages using:

```sh
pip install numpy keras python-chess stockfish
```

## Usage

1. Download and extract the Stockfish engine from [Stockfish official site](https://stockfishchess.org/download/). Ensure the path to the Stockfish binary is correct in the code.

2. Run the training script:

```sh
python train.py
```

## Model Architecture

Two models are defined in the code:

1. `create_model1`: A deeper model with BatchNormalization layers.
2. `create_model2`: A simpler model without BatchNormalization layers.

You can switch between models by commenting/uncommenting the respective lines in the script.

## Functions and Classes

### `fen_to_vec(fen)`

Converts a FEN string to a vector representation suitable for the neural network.

### `create_model1(shape)`

Creates a deep neural network model with BatchNormalization layers.

### `create_model2(shape)`

Creates a simpler neural network model without BatchNormalization layers.

### `generate_random_fen(games)`

Generates a list of unique FEN strings by playing random legal moves on a chess board.

### `generate_random_input(games)`

Generates input vectors and evaluation scores for a given number of games by using the Stockfish engine.

### `train(games, epochs)`

Trains the model for a specified number of epochs using randomly generated game positions and their Stockfish evaluations.

## Training the Model

To train the model, adjust the `games` and `epochs` variables as needed:

```python
games = 10
epochs = 20000
train(games, epochs)
```

The training progress and model checkpoints will be printed and saved periodically.

## Saving and Loading Models

The model and the set of unique FEN strings are saved every 5 epochs to ensure progress is not lost. You can load a previously saved model by uncommenting the load_model line:

```python
# model = load_model("ai_version1.keras")
```
