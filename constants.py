from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')


# last letter in names from config file correspond to one-letter names given to variables in the problem's description
cities_amount = config["PARAMETERS"].getint('cities_amount_L')
infrastructure_cost = config["PARAMETERS"].getint('infrastructure_cost_b')
max_railways_pieces = config["PARAMETERS"].getint('max_railways_pieces_m')
max_connections = config["PARAMETERS"].getint('max_connections_n')
one_rail_cost = config["PARAMETERS"].getint('one_rail_cost_t')
max_cost = config["PARAMETERS"].getint('max_budget_P')

max_city_size = config["TECHNICAL_CONSTRAINTS"].getint("max_city_size")
max_possible_distance = config["TECHNICAL_CONSTRAINTS"].getint("max_possible_distance")

