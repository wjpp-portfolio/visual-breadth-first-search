# visual-breadth-first-search
A python bfs algorithm demonstrated in a user-created maze

This program first requires you to draw a maze and set a start and end location for the search.  The program will then highlight the route from the start to end in yellow.  An impossible route will reset the grid as there is nothing to show.

## controls
Use **arrow keys** to place a node (square) on the grid.  Valid routes between nodes are indicated by a thin edge (line).

press **d** to delete a node and all connected edges.  This also 'lifts' the drawing pen from the grid.  Press **space bar** to put the pen back down

press **e** to place an end marker.  This is where the location the algorithm will attempt to find

press **s** to place a start marker which is where the algorithm will search from

press **space bar** to 'lift' the drawing pen from the page to navigate without drawing nodes.  Press space again to start drawing again

press **return** (enter) to start the breadth first search algorithm

press **r** to reset the entire grid
