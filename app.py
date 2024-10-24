from flask import Flask, render_template, request, session, url_for, redirect
import engine  # to calculate the genetic algorithm
app = Flask(__name__)

app.secret_key = "Ale sekret ten kij"


@app.route('/submit_cities_amount', methods=['POST'])
def submit_cities_amount():
    if request.form['cities_amount'] is None:
        return redirect(url_for("welcome_page"))
    cities_amount = int(request.form['cities_amount'])
    session['cities_amount'] = cities_amount
    return redirect(url_for("submit_cords_page"))

@app.route('/submit_cords', methods=["GET"])
def submit_cords_page():
    if 'cities_amount' not in session:
        return redirect(url_for("welcome_page"))
    return render_template("submit_coordinates.html", cities_amount=session['cities_amount'])
@app.route('/submit_cords', methods=['POST'])
def submit_coordinates():
    if 'cities_amount' not in session:
        return redirect(url_for("welcome_page"))
    coordinates = []
    for i in range(session['cities_amount']):
        x = int(request.form[f'city{i}_x'])
        y = int(request.form[f'city{i}_y'])
        coordinates.append([x, y])
    return (f"<h1>Thank you for submitting the coordinates, further functions in development :)</h1>"
            f"<a href='/'><button>Go to Home</button></a>")


@app.route('/')
def welcome_page():
    return render_template("landing_page.html")

