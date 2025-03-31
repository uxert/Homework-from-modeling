from configparser import ConfigParser

from flask import Flask, render_template, request, session, url_for, redirect
from numpy import array, uint32
import numpy as np
from optimization_engine import EngineWrapper

app = Flask(__name__)

app.secret_key = "Ale sekret ten kij"


@app.route('/submit_cities_amount', methods=['POST'])
def submit_cities_amount():
    if request.form['cities_amount'] is None:
        return redirect(url_for("welcome_page"))

    cities_amount = int(request.form['cities_amount'])
    session['cities_amount'] = cities_amount
    inputted_seed = int(request.form['seed'])
    session['seed'] = None if inputted_seed == 0 else inputted_seed
    parameters = ['infrastructure_cost', 'max_railways_pieces', 'max_connections', 'one_rail_cost', 'max_budget',
                  'max_city_size', 'max_possible_coordinate']
    for param in parameters:
        session[param] = int(request.form[param])
    return redirect(url_for("submit_cords_page"))


@app.route('/submit_cords', methods=["GET"])
def submit_cords_page():
    if 'cities_amount' not in session:
        return redirect(url_for("welcome_page"))

    rand_cords, rand_sizes = EngineWrapper.generate_random_city_cords(session['cities_amount'], session['max_city_size'],
                                                            session['max_possible_coordinate'], session['seed'])
    return render_template("submit_coordinates.html", cities_amount=session['cities_amount'],
                           random_cords=rand_cords, random_sizes=rand_sizes)

@app.route('/submit_cords', methods=['POST'])
def submit_coordinates():
    if 'cities_amount' not in session:
        return redirect(url_for("welcome_page"))
    coordinates = []
    sizes = []
    for i in range(session['cities_amount']):
        x = int(request.form[f'city{i}_x'])
        y = int(request.form[f'city{i}_y'])
        size = int(request.form[f'city{i}_size'])
        coordinates.append([x, y])
        sizes.append(size)
    coordinates = array(coordinates, dtype=uint32)
    sizes = array(sizes, dtype=uint32)
    session['coordinates_ready'] = True
    seed = int(request.form['seed'])
    seed = None if seed == 0 else seed
    algorithm_instance = EngineWrapper(session['cities_amount'], max_city_size=session['max_city_size'],
        max_coordinate_val=session['max_possible_coordinate'], max_cost=session['max_budget'],
        max_railway_len=session['max_railways_pieces'], max_connections_count=session['max_connections'],
        one_rail_cost=session['one_rail_cost'], infrastructure_cost=session['infrastructure_cost'])

    distances = algorithm_instance.calculate_distances_matrix(coordinates, use_manhattan_metric=True)
    print("generated distances")
    last_population = algorithm_instance.genetic_algorithm(distances_matrix=distances, sizes_vector=sizes,
                                                           seed=seed, silent=True)
    last_population_scores = algorithm_instance.goal_function_convenient(distances, sizes, last_population)
    best_index = np.argmax(last_population_scores)
    best_score = last_population_scores[best_index]
    best_solution = last_population[best_index]
    connections = []
    for i in range(session['cities_amount']):
        temp = []
        for j in range(session['cities_amount']):
            if best_solution[i,j] == True:  # == instead of 'is', because numpy apparently does not store them as
                # actual python booleans - checked, using 'is' does not work
                temp.append(j)
        connections.append(temp)
    return render_template("display_results.html", best_score=best_score,
                           best_solution_connections=connections)

@app.route("/display_result", methods=["GET"])
def display_result():
    best_score = session.get("best_score", None)
    if best_score is None:
        return redirect(url_for("welcome_page"))  # create a page with an error in the future




@app.route('/')
def welcome_page():
    my_config = ConfigParser()
    my_config.read("config.ini")
    return render_template("landing_page.html", config_values=my_config)

def run_the_app(use_debug_mode=True):
    """Runs the Flask app, basically the same effect as if you just ran this script"""
    app.run(debug=use_debug_mode)

if __name__ == "__main__":
    run_the_app()