""" This file largely follows the steps outlined in the Insight Flask tutorial, except data is stored in a
flat csv (./assets/births2012_downsampled.csv) vs. a postgres database. If you have a large database, or
want to build experience working with SQL databases, you should refer to the Flask tutorial for instructions on how to
query a SQL database from here instead.

May 2019, Donald Lee-Brown
"""

from flask import render_template
from flaskexample import app
from flaskexample.a_model import ModelIt
import pandas as pd
from flask import request
import test

"""
model_service = load_model('./flaskexample/static/data/service-lstm-weighted.h5')
model_room = load_model('./flaskexample/static/data/lstm-weighted-room.h5')
model_loc = load_model('./flaskexample/static/data/lstm-weighted-location.h5')
"""
# here's the homepage
@app.route('/')
def homepage():
    return render_template("bootstrap_template.html")

# example page for linking things
@app.route('/example_linked')
def linked_example():
    return render_template("example_linked.html")

#here's a page that simply displays the births data
@app.route('/example_dbtable')
def birth_table_page():
    births = []
    # let's read in the first 10 rows of births data - note that the directory is relative to run.py
    dbname = './flaskexample/static/data/births2012_downsampled.csv'
    births_db = pd.read_csv(dbname).head(10)
    # when passing to html it's easiest to store values as dictionaries
    for i in range(0, births_db.shape[0]):
        births.append(dict(index=births_db.index[i], attendant=births_db.iloc[i]['attendant'],
                           birth_month=births_db.iloc[i]['birth_month']))
    # note that we pass births as a variable to the html page example_dbtable
    return render_template('/example_dbtable.html', births=births)

# now let's do something fancier - take an input, run it through a model, and display the output on a separate page

@app.route('/model_input')
def birthmodel_input():
   return render_template("model_input.html")

@app.route('/model_output')
def birthmodel_output():
   # pull 'birth_month' from input field and store it
   x = request.args.get('birth_month')

   # read in our csv file
   the_result = 'lexington ave is a street.'
   service, room, loc, comments = ModelIt(x)
   return render_template("model_output.html", service=service, room=room, loc=loc, comments = comments)
