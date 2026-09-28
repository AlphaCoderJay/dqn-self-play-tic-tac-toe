import random


class TicTacToeEnv:
    def __init__(self,render=False):
        self.board =[0,0,0,0,0,0,0,0,0]

        self.render = render

        if random.random() < 0.5:
            self.current_player = -1

        else:
            self.current_player = 1

    def reset(self):
        self.board =[0,0,0,0,0,0,0,0,0]

        if random.random() < 0.5:
            self.current_player = -1

        else:
            self.current_player = 1
        return self.get_observation(), {}


    #------------WILL DO----------------
    def get_observation(self):
        board = self.board.copy()
        return [b * self.current_player for b in board]
        

    
    
    def ConstBoard(self,board):

        if self.render:
            print("Current State of the Board : \n\n")
            for i in range (0,9):
                if((i>0) and (i%3)==0):
                    print("\n")

                if (board[i] == 0):
                    print("- ", end=" ")

                if (board[i]==1):
                    print("O ", end=" ")

                if (board[i] ==-1):
                    print("X", end=" ")

            print("\n\n")
    

    def step(self,action):
        current_player_reward = 0.0
        terminated = False

        if (self.board[action]!=0):
                current_player_reward = -1
                terminated = True

                observation = self.get_observation()
                return observation , current_player_reward, terminated, False, {}

        #So we need 9 possible actions
        else:
            self.board[action]=self.current_player


        x = self.analyzeboard(self.board)

        #Win
        if (x==self.current_player):
            self.ConstBoard(self.board)
            current_player_reward += 1
            terminated = True
            observation = self.get_observation()
            return observation , current_player_reward, terminated, False, {}

        if (x==0 and 0 not in self.board):
            self.ConstBoard(self.board)
            current_player_reward += 0
            terminated = True
            observation = self.get_observation()
            return observation , current_player_reward,terminated, False, {}


        else:
            terminated = False


        self.current_player *= -1
        observation = self.get_observation()
        return observation, current_player_reward, terminated, False, {}
        





    #This Function is used to analyze a game, i dont get this function we idk the code
    def analyzeboard(self,board):
        cb=[[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]] #???

        for i in range(0,8):
            if (board[cb[i][0]] != 0 and
                board[cb[i][0]] == board[cb[i][1]] and
                board[cb[i][0]] == board[cb[i][2]]):

                return board[cb[i][2]]


        return 0