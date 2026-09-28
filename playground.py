from dqn import DQN
from newenv import TicTacToeEnv
from agent import Agent
import torch
import random
import numpy as np

env = TicTacToeEnv(render=False)
MODEL_PATH = "testmine.pt"
agent = Agent()
agent.online_net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
agent.online_net.eval()


def act_masked_eps(agent, board, dqn_mark, epsilon=0.0):
    """board: raw board (actual -1/0/1 values). dqn_mark: which value the DQN is playing as."""
    legal = [i for i in range(9) if board[i] == 0]

    if random.random() < epsilon:
        return random.choice(legal)

    # Perspective-normalize to match training: own marks -> 1, opponent -> -1
    state = [cell * dqn_mark for cell in board]

    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = agent.online_net(state_t)[0].clone()

    for i in range(9):
        if board[i] != 0:
            q_values[i] = -float('inf')

    return torch.argmax(q_values).item()


def random_move(board):
    legal = [i for i in range(9) if board[i] == 0]
    return random.choice(legal)


def play_game(dqn_mark, epsilon=0.0):
    """dqn_mark: -1 (X) or 1 (O). X always moves first in tic-tac-toe."""
    env.reset()
    board = env.board          # reference to the actual live board, not get_observation()
    dqn_first = (dqn_mark == -1)

    for i in range(9):
        if env.analyzeboard(board) != 0:
            break
        if 0 not in board:
            break

        dqn_turn = ((i % 2 == 0) == dqn_first)

        if dqn_turn:
            action = act_masked_eps(agent, board, dqn_mark, epsilon)
            board[action] = dqn_mark
        else:
            action = random_move(board)
            board[action] = -dqn_mark

    x = env.analyzeboard(board)
    if x == dqn_mark:
        return "win"
    elif x == -dqn_mark:
        return "loss"
    else:
        return "draw"


def run_eval(num_games_per_side=500, epsilon=0.0):
    results = {
        "X": {"win": 0, "loss": 0, "draw": 0},
        "O": {"win": 0, "loss": 0, "draw": 0},
    }

    for _ in range(num_games_per_side):
        outcome = play_game(dqn_mark=-1, epsilon=epsilon)  # DQN plays X
        results["X"][outcome] += 1

    for _ in range(num_games_per_side):
        outcome = play_game(dqn_mark=1, epsilon=epsilon)   # DQN plays O
        results["O"][outcome] += 1

    total_games = num_games_per_side * 2

    for side in ["X", "O"]:
        w, l, d = results[side]["win"], results[side]["loss"], results[side]["draw"]
        n = num_games_per_side
        print(f"\n===== DQN as {side} ({n} games) =====")
        print(f"Wins:   {w} ({w/n:.2%})")
        print(f"Draws:  {d} ({d/n:.2%})")
        print(f"Losses: {l} ({l/n:.2%})")

    total_w = results["X"]["win"] + results["O"]["win"]
    total_l = results["X"]["loss"] + results["O"]["loss"]
    total_d = results["X"]["draw"] + results["O"]["draw"]

    print(f"\n===== Combined ({total_games} games) =====")
    print(f"Wins:   {total_w} ({total_w/total_games:.2%})")
    print(f"Draws:  {total_d} ({total_d/total_games:.2%})")
    print(f"Losses: {total_l} ({total_l/total_games:.2%})")

    return results


run_eval(num_games_per_side=500, epsilon=0)