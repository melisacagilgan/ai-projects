from copy import deepcopy

BLACK = 1
WHITE = 2
EMPTY = 0

MAX = 1000
MIN = -1000

DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1),
              (1, 1), (1, -1), (-1, 1), (-1, -1)]


def opposite(player):
    if (player == BLACK):
        return WHITE
    elif (player == WHITE):
        return BLACK


def add(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1])


class Board:
    def __init__(self):
        '''
            Initializes the board
        '''
        self.__board = [[EMPTY for x in range(8)] for y in range(8)]
        self.__board[3][3] = WHITE
        self.__board[3][4] = BLACK
        self.__board[4][3] = BLACK
        self.__board[4][4] = WHITE

    @staticmethod
    def get_unique_moves(moves):
        '''
            Returns a list of unique moves
        '''
        unique_moves = set([])
        for move in moves:
            (start, direction, end) = move
            unique_moves.add(end)

        return list(unique_moves)

    def get_all_valid_moves(self, color):
        '''
            Returns all valid moves for a given color
        '''
        all_discs = self.__get_all_discs(color)
        valid_moves = []

        for disc in all_discs:
            valid_moves.extend(self.__calculate_move(disc))

        return valid_moves

    def get_number_of_discs(self, color):
        '''
            Returns the number of discs of the given color.
        '''
        counter = 0
        for y in range(8):
            for x in range(8):
                if (self.__board[y][x] == color):
                    counter += 1
        return counter

    def flip(self, moves, target, color):
        '''
            flips discs given moves and end_move
        '''
        for m in moves:
            (start, direction, end) = m
            if (end == target):
                while True:
                    start = add(start, direction)
                    self.__set(start, color)
                    if (start == end):
                        break

    def has_legal_move(self, color):
        all_discs = self.__get_all_discs(color)
        for disc in all_discs:
            moves = self.__calculate_move(disc)
            if (len(moves) > 0):
                return True

        return False

    def __get_all_discs(self, color):
        '''
            Returns all disc position given the color
        '''
        discs = []
        for y in range(8):
            for x in range(8):
                if self.__board[y][x] == color:
                    discs.append((x, y))
        return discs

    def __in_boundary(self, position):
        '''
            Returns True if the given position is in the board
        '''
        (x, y) = position
        if (x < 0 or x > 7 or y < 0 or y > 7):
            return False
        return True

    def __calculate_move(self, start):
        '''
            Returns all valid moves for a given disc
        '''
        color = self.__get(start)

        moves = []

        for direction in DIRECTIONS:
            base = add(start, direction)

            if (self.__in_boundary(base) == False):
                continue

            if (self.__get(base) == EMPTY or self.__get(base) == color):
                continue

            end = start

            while True:
                end = add(end, direction)

                if (self.__in_boundary(end) == False):
                    break

                if (self.__get(end) == color):
                    break

                if (self.__get(end) == EMPTY):
                    moves.append((start, direction, end))
                    break

        return moves

    def __get(self, position):
        '''
            Returns the disc color at a given position
        '''
        (x, y) = position
        if (x < 0 or x > 7 or y < 0 or y > 7):
            return None
        return self.__board[y][x]

    def __set(self, position, color):
        '''
            Sets the disc color at a given position
        '''
        (x, y) = position
        self.__board[y][x] = color

    def __str__(self):
        result = "  0 1 2 3 4 5 6 7\n"
        for y in range(8):
            result += str(y) + " "
            for x in range(8):
                cell = self.__board[y][x]
                if (cell == EMPTY):
                    result += ". "
                elif (cell == BLACK):
                    result += "b "
                elif (cell == WHITE):
                    result += "w "
            result += "\n"
        return result


def ai(board, player, height, heuristic, heuristic_num, output):
    moves = board.get_all_valid_moves(player)
    unique_moves = board.get_unique_moves(moves)

    bestMove = None

    isMax = None

    if (player == BLACK):
        isMax = True
    elif (player == WHITE):
        isMax = False

    if (output):
        if(player == 1):
            print("BLACK's turn (playing w/Heuristic {})".format(heuristic_num))
        else:
            print("WHITE's turn (playing w/Heuristic {})".format(heuristic_num))
        print(board)

    if (len(unique_moves) == 0):
        if (output):
            if(player == 1):
                print("No moves available for BLACK")
            else:
                print("No moves available for WHITE")
        return None

    bestMove = None

    for move in unique_moves:
        board_copy = deepcopy(board)
        board_copy.flip(moves, move, player)
        h, node_count = min_max(
            board_copy, height, player, isMax, heuristic, MIN, MAX)
        if (bestMove == None or h > bestMove[0]):
            bestMove = (h, move, node_count)

    board.flip(moves, bestMove[1], player)
    if (output):
        print("Best move's node count: {}\n--------------------------".format(node_count))


def player(board, player):
    if(player == 1):
        print("BLACK's turn")
    else:
        print("WHITE's turn")
    print(board)

    moves = board.get_all_valid_moves(player)
    unique_moves = board.get_unique_moves(moves)

    if (len(unique_moves) == 0):
        print("For player {}, no moves available".format(player))
        return None

    for i, move in enumerate(unique_moves):
        print(i, move)

    selection = int(input("Select a move: "))
    move = unique_moves[selection]
    board.flip(moves, move, player)


def min_max(board, height, player, isMax, heuristic, alpha, beta, node_count=0):
    '''
        Returns the best move for the player
    '''

    if (height == 0 or board.has_legal_move(player) == False):
        return heuristic(board), 1

    moves = board.get_all_valid_moves(player)
    unique_moves = Board.get_unique_moves(moves)

    best = None

    if (isMax):
        best = MIN
    else:
        best = MAX

    total_node = 0

    for move in unique_moves:
        board_copy = deepcopy(board)
        board_copy.flip(moves, move, player)
        h, node_count = min_max(board_copy, height - 1,
                                opposite(player), not isMax, heuristic, alpha, beta, node_count + 1)
        total_node += node_count

        if (isMax):
            best = max(best, h)
            alpha = max(alpha, best)
        else:
            best = min(best, h)
            beta = min(beta, best)

        if (beta <= alpha):
            break

    return best, total_node


if __name__ == "__main__":
    board = Board()
