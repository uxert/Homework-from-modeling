from configparser import ConfigParser

from flask import Flask, render_template, request, session, url_for, redirect
import engine  # to calculate the genetic algorithm

from engine import EngineWrapper

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
    return (f"<h1>Thank you for submitting the coordinates, further functions in development :)</h1>"
            f"<a href='/'><button>Go to Home</button></a>")


@app.route('/')
def welcome_page():
    my_config = ConfigParser()
    my_config.read("config.ini")
    return render_template("landing_page.html", config_values=my_config)

