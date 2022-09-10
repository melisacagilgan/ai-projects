
class _Node:
    def __init__(self, parent, position):
        self.f = 0
        self.h = 0
        self.g = 0
        self.position = position
        self.parent = parent

    def __eq__(self, other):
        return self.position == other.position


class _Waypoint():
    def __init__(self, label, position):
        self.label = label
        self.position = position


def manhattan_distance(start, end):
    x1 = start[0]
    y1 = start[1]
    x2 = end[0]
    y2 = end[1]
    return (abs(x1 - x2) + abs(y1 - y2))


class Map:
    def __init__(self, mapPath):
        self.map = []
        self.waypoints = []
        self.width = 0
        self.height = 0

        with open(mapPath, "r") as file:
            for i, line in enumerate(file.readlines()):
                row = []
                for j, character in enumerate(line.strip()):
                    position = (j, i)
                    if (character == "*" or character == " "):
                        row.append(character)
                    else:
                        self.waypoints.append(
                            _Waypoint(label=character, position=position))
                        row.append(" ")

                self.map.append(row)

        self.height = len(self.map)
        self.width = len(self.map[0])

    def search(self, start, end, heuristic):
        start_node = _Node(None, start.position)
        end_node = _Node(None, end.position)

        open_list = [start_node]
        closed_list = []

        while (len(open_list) > 0):
            # Finding min node
            min_node = open_list[0]
            for node in open_list:
                if node.f < min_node.f:
                    min_node = node

            open_list.remove(min_node)
            closed_list.append(min_node)

            if (min_node == end_node):
                path = []
                current = min_node
                while current:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]

            # Getting successors of min node
            successors = self.__successors(min_node)
            for successor in successors:
                if (successor in closed_list):
                    continue

                for node in open_list:
                    if (node == successor and successor.g > node.g):
                        continue

                successor.g = min_node.g + 1
                successor.h = heuristic(start.position, end.position)
                successor.f = successor.g + successor.h

                open_list.append(successor)

        return 0

    def __successors(self, node):
        successors = []
        for offset in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            # Calculating new position
            new_position = (offset[0] + node.position[0],
                            offset[1] + node.position[1])
            x = new_position[0]
            y = new_position[1]

            # Checking boundries
            if (x < 0 or y < 0 or (self.width - 1) < x or (self.height - 1) < y):
                continue

            # Checking if given is walkable or not
            if (not self.map[y][x] != "*"):
                continue

            successor = _Node(node, new_position)
            successors.append(successor)

        return successors
