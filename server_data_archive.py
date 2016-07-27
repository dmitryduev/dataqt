"""
    Flask-based server for the Robo-AO Data Archive

    Dr Dmitry A. Duev @ Caltech, 2016
"""

from __future__ import print_function
import os
import json
import datetime
import ConfigParser
import inspect
from collections import OrderedDict
import flask
import flask_login
from werkzeug.security import generate_password_hash, check_password_hash


''' load config data '''
abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
config = ConfigParser.RawConfigParser()
config.read(os.path.join(abs_path, 'config.ini'))


app = flask.Flask(__name__)
app.secret_key = 'roboaokicksass'

login_manager = flask_login.LoginManager()

login_manager.init_app(app)

# Our mock database -> replace with pymongo query to the mongodb database
users = {'admin': {'password': generate_password_hash('robo@0'),
                   'programs': ['0', '3', '4', '5']},
         'user': {'password': generate_password_hash('test'),
                  'programs': ['4']}
         }
# users = {'admin': {'password': 'pbkdf2:sha1:1000$tytMHt8x$121b8c130d98997228c100b13aa82acc9696c172'}}


''' handle user login'''


class User(flask_login.UserMixin):
    pass


@login_manager.user_loader
def user_loader(username):
    if username not in users:
        return

    user = User()
    user.id = username
    return user


@login_manager.request_loader
def request_loader(request):
    username = request.form.get('username')
    if username not in users:
        return

    user = User()
    user.id = username

    user.is_authenticated = check_password_hash(users[username]['password'],
                                                flask.request.form['password'])

    return user


@app.route('/login', methods=['GET', 'POST'])
def login():
    if flask.request.method == 'GET':
        # logged in already?
        if flask_login.current_user.is_authenticated:
            return flask.redirect(flask.url_for('root'))
        # serve template if not:
        else:
            return flask.render_template('template-login.html')
    # print(flask.request.form['username'], flask.request.form['password'])

    username = flask.request.form['username']
    if check_password_hash(users[username]['password'], flask.request.form['password']):
        user = User()
        user.id = username
        flask_login.login_user(user, remember=True)
        return flask.redirect(flask.url_for('root'))
    else:
        # serve template with flag failed=true, display fail message
        return flask.redirect(flask.url_for('login'))


# serve root
@app.route('/')
@flask_login.login_required
def root():
    return 'Logged in as: {:s}'.format(flask_login.current_user.id) + \
        '''
            <form action='logout' method='POST'>
                <input type='submit' name='logout'></input>
            </form>
        '''


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    flask_login.logout_user()
    return flask.redirect(flask.url_for('root'))


@login_manager.unauthorized_handler
def unauthorized_handler():
    return flask.redirect(flask.url_for('login'))


if __name__ == '__main__':
    app.run(host=config.get('Server', 'host'), port=config.get('Server', 'port'), threaded=True)
