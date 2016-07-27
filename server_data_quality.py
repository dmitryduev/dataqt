"""
    Flask-based server for the Robo-AO DQM service

    Dmitry A. Duev @ Caltech, 2016
"""

from __future__ import print_function
import os
import json
import datetime
import ConfigParser
import inspect
from collections import OrderedDict
from flask import Flask, render_template, request, jsonify  # send_from_directory
from flask_basicauth import BasicAuth

app = Flask(__name__)

# basic HTTP authentification
app.config['BASIC_AUTH_USERNAME'] = 'admin'
app.config['BASIC_AUTH_PASSWORD'] = 'robo@0'
basic_auth = BasicAuth(app)

# load config data
abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
config = ConfigParser.RawConfigParser()
config.read(os.path.join(abs_path, 'config.ini'))

# seeing data dwelling place:
# path_to_website_data = config.get('Path', 'path_to_website_data')
path_to_website_data = os.path.join(abs_path, 'static', 'data')
# create if absent:
if not os.path.exists(path_to_website_data):
    os.mkdir(path_to_website_data)


# serve root
@app.route('/')
@basic_auth.required
def hw():
    # get all available dates to display in the calendar
    dates = sorted([lf for lf in os.listdir(path_to_website_data)
                    if os.path.isdir(os.path.join(path_to_website_data, lf))])
    return render_template('template-main.html', dates=dates)


@app.route('/_get_fits_header')
def get_fits_header():
    """
        Get FITS header for a _source_ observed on a _date_
    :return: jsonified dictionary with the header / empty dict if failed
    """
    # get parameters from the AJAX GET request
    date = request.args.get('date', 0, type=str)
    source = request.args.get('source', 0, type=str)

    # load JSON file with the date's data
    f_json_sci = os.path.join(path_to_website_data, '{:s}'.format(date), '{:s}.json'.format(date))

    if os.path.exists(f_json_sci):
        with open(f_json_sci) as fjson_sci:
            data = json.load(fjson_sci, object_pairs_hook=OrderedDict)
            data = OrderedDict(data)
    else:
        # if failed:
        return jsonify(result={})

    # we're displaying images only of the sources for which the pipeline did not fail
    for pot in ('high_flux', 'faint'):
        try:
            # check the full name, as the same source could be observed multiple times in a night
            header = [s['header'] for s in data[pot] if source in s['header']['FILENAME'][0]]
            if len(header) > 0:
                return jsonify(result=OrderedDict(header[0]))
        except:
            return jsonify(result={})

    # if failed:
    return jsonify(result={})


# serve particular date
@app.route('/<int:date>')
@basic_auth.required
def show_date(date):
    # load JSON file summarizing the date's data
    f_json_sci = os.path.join(path_to_website_data, '{:d}'.format(date), '{:d}.json'.format(date))
    # load JSON file summarizing the date's seeing data
    f_json_seeing = os.path.join(path_to_website_data, '{:d}'.format(date), '{:d}.seeing.json'.format(date))

    # previous and next dates for website navigation bar:
    date_m1 = datetime.datetime.strftime(datetime.datetime.strptime(str(date), '%Y%m%d') -
                                         datetime.timedelta(days=1), '%Y%m%d')
    date_p1 = datetime.datetime.strftime(datetime.datetime.strptime(str(date), '%Y%m%d') +
                                         datetime.timedelta(days=1), '%Y%m%d')

    if not os.path.exists(f_json_sci) and not os.path.exists(f_json_seeing):
        return render_template('template-no-data.html', date_m1=date_m1, date_p1=date_p1)
    else:
        # sci data
        if os.path.exists(f_json_sci):
            with open(f_json_sci) as fjson_sci:
                # preserve sorting order from the JSON file
                data = json.load(fjson_sci, object_pairs_hook=OrderedDict)
                data = OrderedDict(data)
        else:
            data = False
        # seeing
        if os.path.exists(f_json_seeing):
            with open(f_json_seeing) as fjson_seeing:
                seeing = json.load(fjson_seeing, object_pairs_hook=OrderedDict)
                seeing = OrderedDict(seeing)
        else:
            seeing = False

        return render_template('template.html', date=str(date), data=data, seeing=seeing,
                               date_m1=date_m1, date_p1=date_p1)


if __name__ == '__main__':
    app.run(host=config.get('Server', 'host'), port=config.get('Server', 'port'), threaded=True)
