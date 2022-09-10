from collections import deque
from queue import PriorityQueue


class _Path():
    def __init__(self, current, visited):
        self.current = current
        self.visited = visited

    def __lt__(self, other):
        return self.current < other.current


class Graph:
    def __init__(self, map, heuristic):
        self.graph = {}
        for point_A in map.waypoints:
            self.graph[point_A.label] = {}
            for point_B in map.waypoints:
                if (point_A != point_B):
                    path = map.search(point_A, point_B, heuristic)
                    self.graph[point_A.label][point_B.label] = len(path)

    def bfs(self, start):
        queue = deque([_Path(start, [])])

        while (len(queue[0].visited) != len(self.graph) - 1):
            path = queue.popleft()
            for node in self.graph[path.current]:
                if (node not in path.visited):
                    queue.append(_Path(node, path.visited + [path.current]))

        path = queue[0]
        return path.visited + [path.current, start]

    def ucs(self, start):
        current_path = _Path("A", [])
        queue = PriorityQueue()
        queue.put((0, current_path))

        while (len(current_path.visited) != len(self.graph) - 1):
            weight, current_path = queue.get()
            graphDictionary = self.graph[current_path.current]

            for node in graphDictionary:
                localWeight = weight
                if (node not in current_path.visited):
                    localWeight += self.graph[current_path.current][node]
                    history = current_path.visited + [current_path.current]
                    queue.put((localWeight, _Path(node, history)))

        return current_path.visited + [current_path.current, start]

    def cost(self, path):
        cost = 0
        for i in range(len(path) - 1):
            currentNode = path[i]
            nextNode = path[i + 1]
            cost += self.graph[currentNode][nextNode]
        return cost

    def __str__(self):
        result = ""
        for A in sorted(self.graph):
            for B in sorted(self.graph[A]):
                if (A != B):
                    result += "{},{},{}\n".format(A, B, self.graph[A][B])
        return result
