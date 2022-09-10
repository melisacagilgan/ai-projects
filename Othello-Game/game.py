from board import Board, player, ai, BLACK, WHITE
from time import time
import random


def heuristic_1(board):
    return board.get_number_of_discs(BLACK) - board.get_number_of_discs(WHITE)


def heuristic_2(board):
    white_moves = board.get_all_valid_moves(WHITE)
    black_moves = board.get_all_valid_moves(BLACK)
    return len(black_moves) - len(white_moves)


class Game:
    def __init__(self, mode):
        self.__mode = mode
        self.__board = Board()

    def new_game(self):
        self.__board = Board()

    def start(self, height=3, heuristic_1=heuristic_1, heuristic_2=heuristic_2, output=True):
        start = time()
        disc_list = [BLACK, WHITE]
        random_disc = random.choice(disc_list)
        while True:
            black_moves = self.__board.has_legal_move(color=BLACK)
            white_moves = self.__board.has_legal_move(color=WHITE)

            if (not black_moves and not white_moves):
                if (output):
                    print("Neither can move terminating game")
                break

            if (self.__mode == "player"):
                if(random_disc == BLACK):
                    player(self.__board, BLACK)
                    ai(self.__board, WHITE, height, heuristic_1, 1, output)
                else:
                    ai(self.__board, BLACK, height, heuristic_1, 1, output)
                    player(self.__board, WHITE)
            elif (self.__mode == "computer"):
                if(random_disc == BLACK):
                    ai(self.__board, BLACK, height, heuristic_1, 1, output)
                    ai(self.__board, WHITE, height, heuristic_2, 2, output)
                else:
                    ai(self.__board, BLACK, height, heuristic_2, 2, output)
                    ai(self.__board, WHITE, height, heuristic_1, 1, output)

        end = time()

        if (output):
            print(self.__board)

        game_time = end - start

        no_black = self.__board.get_number_of_discs(BLACK)
        no_white = self.__board.get_number_of_discs(WHITE)

        if (no_black == no_white):
            winner = "Draw"
        elif (no_black > no_white):
            winner = "Black"
        else:
            winner = "White"
        return random_disc, game_time, no_black, no_white, winner


if __name__ == "__main__":
    game_dict = {}
    h1_winner = [0, 0]
    h2_winner = [0, 0]
    game_count = 0
    game_time = 0
    no_black = 0
    no_white = 0
    winner = ""
    random_disc = ""

    game_mode = input("Enter gamemode (player|computer): ")
    if (game_mode == "computer"):
        while game_count < 100:
            depth = random.randint(0, 3)
            game_number = "Game "+str(game_count+1)
            game = Game(mode="computer")
            random_disc, game_time, no_black, no_white, winner = game.start(height=depth, heuristic_1=heuristic_1,  heuristic_2=heuristic_2,
                                                                            output=False)
            if random_disc == BLACK:
                if winner == 'Black':
                    h1_winner[0] += 1
                else:
                    h2_winner[1] += 1
            else:
                if winner == 'Black':
                    h2_winner[0] += 1
                else:
                    h1_winner[1] += 1

            game_dict = {game_number: {'Number of predicted moves': depth, 'Time': game_time,
                                       'Black\'s Points': no_black, 'White\'s Points': no_white, 'Winner':  winner}}

            game_count += 1
            print(game_dict)
        print("Black: {} and White: {} times won w/heuristic 1\nBlack: {} and White: {} times won w/heuristic 2".format(
            h1_winner[0], h1_winner[1], h2_winner[0], h2_winner[1]))
    elif (game_mode == "player"):
        game = Game(mode="player")
        _, game_time, no_black, no_white, winner = game.start()
        game_dict = {'Time': game_time,
                     'Black\'s Points': no_black, 'White\'s Points': no_white, 'Winner':  winner}
        print(game_dict)
    else:
        print("Invalid selection")
