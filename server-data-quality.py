import os
import json
# import ConfigParser
import inspect
from flask import Flask, render_template, send_from_directory
app = Flask(__name__)

# load config data
abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
# config = ConfigParser.RawConfigParser()
# config.read(os.path.join(abs_path, 'config.ini'))

# seeing data dwelling place:
# path_to_seeing_data = config.get('Path', 'path_to_seeing_data')
# path_to_website_data = config.get('Path', 'path_to_seeing_data')
path_to_website_data = os.path.join(abs_path, 'data')


# serve static files
@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory(os.path.join(abs_path, 'js'), path)


@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory(os.path.join(abs_path, 'css'), path)


@app.route('/img/<path:path>')
def send_img(path):
    return send_from_directory(os.path.join(abs_path, 'img'), path)


@app.route('/fonts/<path:path>')
def send_fonts(path):
    return send_from_directory(os.path.join(abs_path, 'fonts'), path)


@app.route('/data/<path:path>')
def send_data(path):
    return send_from_directory(os.path.join(abs_path, 'data'), path)


# serve root
@app.route('/')
def hw():
    dates = sorted([lf for lf in os.listdir(path_to_website_data)
                    if os.path.isdir(os.path.join(path_to_website_data, lf))])
    return render_template('template-main.html', dates=dates)


# serve particular date
@app.route('/<int:date>')
def show_date(date):
    f_json_sci = os.path.join(path_to_website_data, '{:d}'.format(date), '{:d}.json'.format(date))
    f_json_seeing = os.path.join(path_to_website_data, '{:d}'.format(date), '{:d}.seeing.json'.format(date))

    if not os.path.exists(f_json_sci) and not os.path.exists(f_json_seeing):
        return render_template('template-no-data.html')
    else:
        # sci data
        if os.path.exists(f_json_sci):
            with open(f_json_sci) as fjson_sci:
                data = json.load(fjson_sci)
                data = dict(data)
        else:
            data = False
        # seeing
        if os.path.exists(f_json_seeing):
            with open(f_json_seeing) as fjson_seeing:
                seeing = json.load(fjson_seeing)
                seeing = dict(seeing)
        else:
            seeing = False
        return render_template('template.html', date=str(date), data=data, seeing=seeing)


if __name__ == '__main__':
    app.run(host='localhost', port=8082)
