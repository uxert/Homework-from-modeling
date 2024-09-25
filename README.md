# Homework-from-modeling
This repo is a project solving my homework from Mathematic Modeling - one of my subjects in university. This project is meant to (eventually...) be a full-stack app.

## Simplified (for now) problem to solve:

Your boss has become very interested in a game strikingly similar to Transport Tycoon, where various transport services are provided to earn money. The boss is exclusively interested in train transport because he read online that it is the most profitable activity at the beginning of the game.

Everything takes place on a map with `L`[^1] cities. At the beginning of the game only cities are present on the map, nothing else.
Each of these cities is separated by a certain distance `d` expressed in arbitrary units called “tiles.” Any two cities can be connected by building exactly `d` pieces of rails and additionaly incurring a fixed infrastructure cost `b`.
Building one piece of track, which allows covering the distance of one tile, costs `t` money. Each city has its own size `g` representing the population of that city.

The amount of money earned from a route between two cities is a function `F` of the size of these cities and the distance between them.

The task given by the boss is: Find a way to connect the cities to earn as much money as possible. The overall cost is not important as long as it is smaller than the budget `P`.

Additionally, the boss, who is a very beginner player and cannot create an advanced railway network, imposed additional constraints:

- Each connection must be direct and only between two cities - to avoid the need to synchronize trains from different routes on the same tracks.
- You can lay down a maximum of `m` pieces of track, where each "piece" covers one unit of distance – so that the boss doesn’t have to build too much.
- At most `n` connections can be created.

There are also some simplifications resulting from the fact that the boss is using a specially modified (simplified at his request) version of the game:

- All connections can always be built in a straight line – only the distance matters.
- You can ignore whether tracks on different routes intersect.
- Each city can be connected to an unlimited number of other cities.
- Different connections in one city do not affect each other in any way.
- The entire investment generates no maintenance costs – only the initial construction cost matters.
- Everything works flawlessly and never breaks down.
- Exactly one train runs on each route - no need for synchronization.

[^1]: Each of these one-letter variables is a parameter given when solving a specific case of the problem.