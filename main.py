import collections
import itertools
import heapq
from collections import defaultdict
from html.parser import HTMLParser


class BenHEAP:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = '<removed-task>'  # placeholder for a removed task
        self.counter = itertools.count()  # unique sequence count

    def add_task(self, task, priority=0):
        """Add a new task or update the priority of an existing task"""
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)

    def remove_task(self, task):
        """Mark an existing task as REMOVED.  Raise KeyError if not found."""
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        """Remove and return the lowest priority task. Raise KeyError if empty."""
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')

    def __contains__(self, item):
        return item in self.entry_finder

    def __bool__(self):
        if self.pq:
            return True
        return False


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def allowed(operation, node, n):
    if ((node.x + operation.x >= 0) and (node.x + operation.x < n)) and \
            ((node.y + operation.y >= 0) and (node.y + operation.y < n)):
        return True
    else:
        return False


def create_board(matrix, node, operation):
    new_matrix = [list(i) for i in matrix]
    # new_matrix = copy.deepcopy(matrix)
    new_node = Point(node.x + operation.x, node.y + operation.y)
    new_matrix[node.x][node.y], new_matrix[new_node.x][new_node.y] = new_matrix[new_node.x][new_node.y], \
                                                                     new_matrix[node.x][node.y]
    return tuple(tuple(i) for i in new_matrix)


def create_all_boards_options(matrix, node, n):
    operation_array = [Point(1, 0), Point(0, 1), Point(-1, 0), Point(0, -1)]
    all_boards_options = []

    for operation in operation_array:
        if allowed(operation, node, n):
            new_m = create_board(matrix, node, operation)
            all_boards_options.append(new_m)

    return all_boards_options


def print_matrix_nicely(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def create_random_mat(node, n_size):
    mat = [[0 for i in range(n_size)] for j in range(n_size)]
    return mat


def generate_keys_for_matrix_size_n(size_n):
    keys = {}
    key_counter = 1
    for x in range(size_n):
        for y in range(size_n):
            keys[key_counter] = Point(x, y)
            key_counter += 1
    return keys


def calculate_nyc_distance(point, should_be_point):
    return abs(point.x - should_be_point.x) + abs(point.y - should_be_point.y)


def calculate_nyc_distance_of_matrix(mat, keys, n):
    nyc_cost = 0
    for x in range(n):
        for y in range(n):
            nyc_cost += calculate_nyc_distance(Point(x, y), keys[mat[x][y]])
    return nyc_cost


def f(h_cost, g_cost):
    return h_cost + g_cost


def h(mat, keys, n):
    return calculate_nyc_distance_of_matrix(mat, keys, n)


def find_special_point(current, sign, n):
    for x in range(n):
        for y in range(n):
            if current[x][y] == sign:
                return Point(x, y)
    raise Exception


def reconstruct_path(cameFrom, current):
    total_path = collections.deque()
    while current in cameFrom.keys():
        current = cameFrom[current]
        total_path.appendleft(current)
    return list(total_path)


def A_star_algo_main(n, special_sign, first_mat, keys):
    INF = 9999999
    openSet = BenHEAP()
    openSet.add_task(first_mat, f(h(first_mat, keys, n), 0))
    counter = 0
    cameFrom = {}

    gScore = defaultdict(lambda: INF)
    gScore[first_mat] = 0

    fScore = defaultdict(lambda: INF)
    fScore[first_mat] = h(first_mat, keys, n)

    while openSet:
        counter += 1
        current = openSet.pop_task()

        if calculate_nyc_distance_of_matrix(current, keys, n) == 0:
            print("FOUND")
            return reconstruct_path(cameFrom, current)

        for possible_mat in create_all_boards_options(current, find_special_point(current, special_sign, n), n):
            tentative_gScore = gScore[current] + 1
            if tentative_gScore < gScore[possible_mat]:
                cameFrom[possible_mat] = current
                gScore[possible_mat] = tentative_gScore
                fScore[possible_mat] = gScore[possible_mat] + h(possible_mat, keys, n)
                if possible_mat not in openSet:
                    openSet.add_task(possible_mat, fScore[possible_mat])
    print("NOT FOUND")


def compare(matA, matB, n):
    for x in range(n):
        for y in range(n):
            if (matA[x][y] != matB[x][y]):
                # print(f"{matA[x][y]} swap with {matB[x][y]}")
                return matA[x][y], matB[x][y]


class MyHTMLParser(HTMLParser):
    def __init__(self, special_sign):
        super().__init__()
        self.g_mat = []
        self.special_sign = special_sign

    def handle_starttag(self, tag, attrs):
        for attr in attrs:
            if attr[0] == "value":
                if attr[1] == " ":
                    self.g_mat.append(self.special_sign)
                else:
                    self.g_mat.append(int(attr[1]))


def parseInputFromFile(file_name, special_sign):
    with open(file_name, 'r') as reader:
        html_input = reader.read()

    parser = MyHTMLParser(special_sign)
    parser.feed(html_input)
    return parser.g_mat


def getMatFromList(val_list, n):
    counter = 0
    mat = [[0 for i in range(n)] for j in range(n)]
    for x in range(n):
        for y in range(n):
            mat[x][y] = val_list[counter]
            counter += 1
    return mat


if __name__ == '__main__':
    file_name = "input.txt"
    n = 3

    # The sign that represents BLANK/HOLE tile
    special_sign = n * n

    # Code if you want to use other examples go to :
    # https://www.mathsisfun.com/games/arrange.html
    # right-click + Inspect on Chrome and right-click on <div id = "board"> + paste it on input.txt file and un comment
    # the code below HAVE FUN
    # val_list = parseInputFromFile(file_name, special_sign)
    # mat = getMatFromList(val_list,n)
    # tuple_mat = tuple(tuple(i) for i in mat)

    example1 = ((9, 3, 2),
                (8,7,1),
                (4,5,6))

    example2 = ((8, 3, 5),
                (6, 9, 1),
                (2, 7, 4))

    example3 = ((7, 2, 9),
                (5, 3, 8),
                (1, 4, 6))

    start = example1

    keys = generate_keys_for_matrix_size_n(n)

    print("First Matrix")
    print_matrix_nicely(start)
    print(f"Manhattan distance is: {calculate_nyc_distance_of_matrix(start, keys, n)}")

    path = A_star_algo_main(n, special_sign, start, keys)
    print(f"Number of min steps you need to solve the puzzle: {len(path)}")

    print("STEPS: ")
    for x in range(0, len(path) - 1):
        valA, valB = compare(path[x], path[x + 1], n)
        if valA != special_sign:
            print(valA)
        else:
            print(valB)
        # print(f"{valA} swap with {valB}")
