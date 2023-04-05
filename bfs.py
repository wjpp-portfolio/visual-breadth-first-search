from contextlib import contextmanager
from typing import List, Tuple, Optional, Set, Iterable

import pygame
from collections import deque


SCREEN_WIDTH = 10
SCREEN_HEIGHT = 10
TRAIL_OFFSET = 6
BLOCK_SIZE = 30
ADJACENCY_LINK_THICKNESS = 6
ROUTE_THICKNESS = 18

WINDOW_SIZE = (BLOCK_SIZE * SCREEN_WIDTH,
               BLOCK_SIZE * SCREEN_HEIGHT)
RECT_DIM = (BLOCK_SIZE, BLOCK_SIZE)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
PINK = (255, 0, 255)
BLUE = (0, 255, 255)
YELLOW = (255, 255, 0)


# class for easier reading of tuple cooordinates
class Coordinate:
    def __init__(self, y: int, x: int):
        self.y = y
        self.x = x

    def __str__(self):
        return f'({self.x}, {self.y})'

    def __repr__(self):
        return str(self)

    def astuple(self) -> Tuple[int, int]:
        return self.y, self.x

    def __iter__(self) -> Iterable[int]:
        yield self.y
        yield self.x

    def __eq__(self, other: 'Coordinate') -> bool:
        return other is not None and self.y == other.y and self.x == other.x

    def copy(self) -> 'Coordinate':
        return Coordinate(self.y, self.x)


VisitMatrix = List[List[Optional[int]]]
AdjMatrix = List[List[Set[Tuple[int, int]]]]


def build_adj_matrix(y: int, x: int) -> AdjMatrix:
    return [
        [set() for _ in range(x)]
        for _ in range(y)
    ]


def build_visit_matrix(y: int, x: int) -> VisitMatrix:
    return [[None] * x for _ in range(y)]


class GridError(Exception):
    pass


class Grid:
    def __init__(self):
        self.drawing = True  # used to define whether nodes are drawn, or the cursor can be moved without drawing
        self.square = Coordinate(0, 0)
        self.start: Optional[Coordinate] = None
        self.end: Optional[Coordinate] = None

        self.array = build_adj_matrix(SCREEN_HEIGHT, SCREEN_WIDTH)
        self.array[self.square.y][self.square.x] = {self.square.astuple()}
        self.solution: Optional[List[Coordinate]] = None

    def pathfind(self):
        self.drawing = False
        self.solution = breadth_first_search(self.array, self.start, self.end)

    def place_start(self):
        self.start = self.square.copy()

    def place_end(self):
        self.end = self.square.copy()

    def delete_node(self):
        self.drawing = False
        sy, sx = self.square
        here = self.array[sy][sx]

        for tup in here:  # iterate through all neighbour adjacencies
            ny, nx = tup
            there = self.array[ny][nx]
            if here is not there:
                there.remove(self.square.astuple())  # from the deleted square from the neighbour

        here.clear()  # remove all neighbour information from deleted square

        # remove start marker if start tile was deleted
        if self.square == self.start:
            self.start = None
        # remove end marker if end tile was deleted
        if self.square == self.end:
            self.end = None

    def switch_pen(self):
        self.drawing = not self.drawing

    def left(self):
        self.square.x = max(self.square.x - 1, 0)

    def right(self):
        self.square.x = min(self.square.x + 1, SCREEN_WIDTH - 1)

    def up(self):
        self.square.y = max(self.square.y - 1, 0)

    def down(self):
        self.square.y = min(self.square.y + 1, SCREEN_HEIGHT - 1)

    @contextmanager
    def move(self):
        prev = self.square.copy()
        yield

        if self.drawing:
            # add an adjacency from this location to previous location
            sy, sx = self.square
            self.array[sy][sx] |= {self.square.astuple(), prev.astuple()}

            # add an adjacency from previous location to this location
            py, px = prev
            self.array[py][px] |= {self.square.astuple()}


def breadth_first_search(graph: AdjMatrix, start: Coordinate, end: Coordinate) -> List[Coordinate]:
    """
    This function recieves a 2d matrix of x/y coordinates where each location contains
    a list of tuples which contain adjacency information.  The function will return
    a list of tuples of the route from start_location to goal or an empty list if
    no route is found. start_location and goal must be tuples in the format (x,y)
    """

    if start is None:
        raise GridError('Start not set')
    if end is None:
        raise GridError('End not set')

    queue = deque([end])  # use queue.popleft() for FIFO queue
    queue2 = []
    pathfind_counter = 0
    visit_matrix = build_visit_matrix(len(graph), len(graph[0]))
    visit_matrix[end.y][end.x] = pathfind_counter

    while len(queue) > 0:
        pathfind_counter += 1
        for i in queue:
            if i == start:  # if location = the start, the search is complete
                counter = 0
                # builds the path back from end to start into a list of tuples
                path = [start]
                for steps in range(visit_matrix[start.y][start.x], 0, -1):
                    py, px = path[counter]
                    for adjacency in graph[py][px]:
                        ny, nx = adjacency
                        if visit_matrix[ny][nx] == steps - 1:
                            counter += 1
                            path.append(Coordinate(*adjacency))
                            break
                return path

            for ny, nx in graph[i.y][i.x]:
                # if neighbour has not been visited, mark it with number, and append its location to the queue
                if visit_matrix[ny][nx] is None:
                    visit_matrix[ny][nx] = pathfind_counter
                    queue2.append(Coordinate(ny, nx))

        queue = list(queue2)
        queue2.clear()

    raise GridError('No path found')


class Display:
    def __init__(self):
        sysfont = pygame.font.get_default_font()
        self.in_square_font = pygame.font.Font(sysfont, 14)
        self.display = pygame.display.set_mode(WINDOW_SIZE)

    def route_lines(self, grid: Grid):
        # print route lines
        if grid.solution:
            for step in range(len(grid.solution) - 1):
                here = grid.solution[step]
                there = grid.solution[step + 1]
                pygame.draw.line(
                    self.display, YELLOW,
                    (
                        (here.x * BLOCK_SIZE) + (BLOCK_SIZE / 2),
                        (here.y * BLOCK_SIZE) + (BLOCK_SIZE / 2),
                    ),
                    (
                        (there.x * BLOCK_SIZE) + (BLOCK_SIZE / 2),
                        (there.y * BLOCK_SIZE) + (BLOCK_SIZE / 2),
                    ),
                    ROUTE_THICKNESS,
                )
        else:
            if grid.drawing:
                colour = BLACK
            else:
                colour = BLUE
            rectpos = grid.square.x * BLOCK_SIZE, grid.square.y * BLOCK_SIZE
            pygame.draw.rect(self.display, colour, ((rectpos), (RECT_DIM)))

    def draw_squares(self, grid: Grid):
        # print squares and adjacencies
        for row in range(SCREEN_HEIGHT):
            for col in range(SCREEN_WIDTH):
                if (row, col) in grid.array[row][col]:
                    pygame.draw.rect(
                        self.display, PINK,
                        pygame.Rect(
                            (col * BLOCK_SIZE) + TRAIL_OFFSET,
                            (row * BLOCK_SIZE) + TRAIL_OFFSET,
                            BLOCK_SIZE - (TRAIL_OFFSET * 2),
                            BLOCK_SIZE - (TRAIL_OFFSET * 2),
                        )
                    )

                    for neighbour in grid.array[row][col]:
                        ny, nx = neighbour

                        pygame.draw.line(
                            self.display, PINK,
                            (
                                (col * BLOCK_SIZE) + (BLOCK_SIZE / 2),
                                (row * BLOCK_SIZE) + (BLOCK_SIZE / 2),
                            ),
                            (
                                (nx * BLOCK_SIZE) + (BLOCK_SIZE / 2),
                                (ny * BLOCK_SIZE) + (BLOCK_SIZE / 2),
                            ),
                            ADJACENCY_LINK_THICKNESS,
                        )

    def draw_icons(self, letter: str, loc: Optional[Coordinate]):
        # draws start icon
        if loc is not None:
            text = self.in_square_font.render(letter, True, BLACK)
            self.display.blit(text, ((loc.x * BLOCK_SIZE) + BLOCK_SIZE * 0.4,
                                     (loc.y * BLOCK_SIZE) + BLOCK_SIZE * 0.25))

    def show(self, grid: Grid):
        self.display.fill(WHITE)
        self.route_lines(grid)
        self.draw_squares(grid)
        self.draw_icons('s', grid.start)
        self.draw_icons('e', grid.end)
        pygame.display.update()


def session(display: Display):
    grid = Grid()
    display.show(grid)

    keys = {
        pygame.K_SPACE: grid.switch_pen,
        pygame.K_LEFT: grid.left,
        pygame.K_RIGHT: grid.right,
        pygame.K_UP: grid.up,
        pygame.K_DOWN: grid.down,
        pygame.K_s: grid.place_start,
        pygame.K_e: grid.place_end,
        pygame.K_d: grid.delete_node,
        pygame.K_RETURN: grid.pathfind,
    }

    while True:
        event = pygame.event.wait()

        if event.type == pygame.QUIT:
            exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                return  # reset

            handler = keys.get(event.key)
            if handler:
                with grid.move():
                    handler()

                display.show(grid)

        if event.type in {
            pygame.ACTIVEEVENT,
            pygame.VIDEOEXPOSE,
            pygame.VIDEORESIZE,
        }:
            display.show(grid)


def main():
    pygame.init()

    try:
        display = Display()

        while True:
            try:
                session(display)
            except GridError as e:
                print(str(e))
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
