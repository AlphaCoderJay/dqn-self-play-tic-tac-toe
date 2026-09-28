from dqn import DQN
from newenv import TicTacToeEnv
from agent import Agent
import torch
AI, HUMAN = -1, 1
env = TicTacToeEnv(render=True)
MODEL_PATH = "testmine.pt"
agent = Agent()
agent.online_net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
agent.online_net.eval()
def act_masked(agent, state, board):
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # shape [1, 9]
        q_values = agent.online_net(state_t)[0]  # back to shape [9]
    q_values = q_values.clone()
    for i in range(9):
        if board[i] != 0:
            q_values[i] = -float('inf')
    return torch.argmax(q_values).item()

def User1Turn(board):
    pos=int(input("Enter O's positions from [0...8]: "))
    if (board[pos]!=0):
        print("Wrong Move!!!")
        exit(0)

    board[pos]=HUMAN

def ai_vs_human():
    board,_ = env.reset()


    print(f"Computer : X Vs. You: O")
    
    player = int(input("Enter to play 1(st) or 2(nd): "))
    for i in range(0,9):
        
        

        if(env.analyzeboard(board) != 0):
            break
        if ((i+player)%2==0):
            state = board
            action = act_masked(agent, state, board)
            board[action] = AI
            print(action, type(action))


        else:
            env.ConstBoard(board)
            User1Turn(board)


    x=env.analyzeboard(board=board)
    if (x==0):
        env.ConstBoard(board)
        print("Draw")

    if (x==HUMAN):
        env.ConstBoard(board)
        print("Computer: Loss, Player: win!")

    if(x==AI):
        env.ConstBoard(board)
        print("Computer: Win, Player: Loss!")



ai_vs_human()

