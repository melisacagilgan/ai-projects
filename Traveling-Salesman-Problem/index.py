import time
from map import Map, manhattan_distance
from graph import Graph

if __name__ == "__main__":

    mapPath = input("Enter your map file: ")
    map = Map(mapPath)
    graph = Graph(map, manhattan_distance)

    action = None

    while action != 3:
        print("-----MENU-----")
        print("Which action would you like to perform ?")
        print("1) Construct a shortest path graph")
        print("2) Solution with BSF and UCS")
        print("3) Exit")

        action = int(input("Action to perform: "))
        while action != 1 and action != 2 and action != 3:
            print("Incorrect output! Please re-enter")
            action = int(input("Action to perform: "))
        if (action == 1):
            print(graph)

        elif (action == 2):
            bfs_start = time.time()
            bfs_path = graph.bfs("A")
            bfs_end = time.time() - bfs_start
            str_bfs_path = "-".join(bfs_path)
            bfs_cost = graph.cost(bfs_path)

            print("Algorithm Used: BFS")
            print(str_bfs_path)
            print("Total Tour Cost: {}\n".format(bfs_cost))

            ucs_start = time.time()
            ucs_path = graph.ucs("A")
            ucs_end = time.time() - ucs_start
            str_ucs_path = "-".join(ucs_path)
            ucs_cost = graph.cost(ucs_path)

            print("Algorithm Used: UCS")
            print(str_ucs_path)
            print("Total Tour Cost: {}\n".format(ucs_cost))

            print("Statistics:")
            print("\tNodes\t\tTime\t\t\tCost")
            print("BFS\t{}\t{}\t{}".format(str_bfs_path, bfs_end, bfs_cost))
            print("UCS\t{}\t{}\t{}".format(str_ucs_path, ucs_end, ucs_cost))
    print("Bye!")
