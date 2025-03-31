# My own genetic algorithm
This is a project solving optimization problem defined for my homework from Mathematic Modeling - one of my subjects in university. 
**Main focus of this project was to implement a genetic algorithm**, because after intro to SI class I was curious about it and wanted to build one myself. 
The optimization problem I had to define on
my Mathematic Modeling was just an excuse to build it really, I wanted to do so for quite some time.

The whole thing is built on numpy, and numpy is basically the only non-standard (although almost everyone considers it a standard at this point...) python library required to run it.

The genetic algorithm I built here was made specifically to solve the problem defined below and it's the only thing it can do. 
The concept of a genetic algorithm remains universal of course, and that's the thing I cared most about anyway when creating this project.

# How to try it out?
Just run the main.py script. It will launch a Flask app that will allow you to give inputs to the algorithm and show you it's output. Nothing fancy and no visualisations (yet...)
but it gets the job done and gives some way to run the genetic algorithm, which was my main concern when creating this project.


# What problem is being solved?
*If you want to know how the optimization problem, that is being solved by this algorithm, was actually derived, I put the whole modeling process in MM_tycon.pdf file. 
It is written in Polish though, as that's the language on my university.*

### Story behind the problem
Your boss has become very interested in a game strikingly similar to Transport Tycoon, where various transport services are provided to earn money. 
The boss is exclusively interested in train transport because he read online that it is the most profitable activity at the beginning of the game.

Everything takes place on a map with `L`[^1] cities. At the beginning of the game only cities are present on the map, nothing else.
Each of these cities is separated by a certain distance `d` expressed in arbitrary units called “tiles.” 
Any two cities can be connected by building exactly `d` pieces of rails and additionaly incurring a fixed infrastructure cost `b`.
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


### Technical constraints
Due to the capabilities of the aforementioned game's engine there are some numerical constraints:
- **total amount of cities cannot be greater than 255**
- max city size and max distance between any 2 cities cannot be larger than 2^32-1
- both infrastructure cost and one railway cost cannot be greater than 2^32 - 1
- money can be calculated up to the value 2^64 - 1 (should be more than enough...)
- max railway pieces amount can be up to 2^64 - 1


### Your task:
Given any map with defined cities sizes and distances, find the best possible way to connect those cities with train rails.

[^1]: Each of these one-letter variables is a parameter given when solving a specific case of the problem and each one
is accessible through config.ini file.

