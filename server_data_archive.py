"""
    Flask-based server for the Robo-AO Data Archive

    Dr Dmitry A. Duev @ Caltech, 2016
"""

from __future__ import print_function
import os
from pymongo import MongoClient
import json
import datetime
import ConfigParser
import sys
import inspect
from collections import OrderedDict
import flask
import flask_login
from werkzeug.security import generate_password_hash, check_password_hash


''' load config data '''
abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
config = ConfigParser.RawConfigParser()
config.read(os.path.join(abs_path, 'config.ini'))
# logger.debug('Successfully read in the config file {:s}'.format(args.config_file))


''' connect to mongodb database '''
# database access:
mongo_host = config.get('Database', 'host')
mongo_port = int(config.get('Database', 'port'))
mongo_db = config.get('Database', 'db')
mongo_user = config.get('Database', 'user')
mongo_pwd = config.get('Database', 'pwd')
mongo_collection_pwd = config.get('Database', 'collection_pwd')

''' Connect to the mongodb database '''
try:
    client = MongoClient(host=mongo_host, port=mongo_port)
    db = client[mongo_db]
    # logger.debug('Successfully connected to the Robo-AO database at {:s}:{:d}'.
    #              format(mongo_host, mongo_port))
except Exception as e:
    print(e)
    # logger.error(e)
    # logger.error('Failed to connect to the Robo-AO database at {:s}:{:d}'.
    #              format(mongo_host, mongo_port))
    # sys.exit()
    db = None
try:
    db.authenticate(mongo_user, mongo_pwd)
    # logger.debug('Successfully authenticated with the Robo-AO database at {:s}:{:d}'.
    #              format(mongo_host, mongo_port))
except Exception as e:
    print(e)
    # logger.error(e)
    # logger.error('Authentication failed for the Robo-AO database at {:s}:{:d}'.
    #              format(mongo_host, mongo_port))
    # sys.exit()
    pass
try:
    coll = db[mongo_collection_pwd]
    # cursor = coll.find()
    # for doc in cursor:
    #     print(doc)
    # logger.debug('Using collection {:s} with user credentials data in the database'.
    #              format(mongo_collection_pwd))
except Exception as e:
    print(e)
    # logger.error(e)
    # logger.error('Failed to use a collection {:s} with user credentials data in the database'.
    #              format(mongo_collection_pwd))
    # sys.exit()
    coll = None


''' initialize the Flask app '''
app = flask.Flask(__name__)
app.secret_key = 'roboaokicksass'

login_manager = flask_login.LoginManager()

login_manager.init_app(app)

''' throw error 500 if could not connect to database '''
if db is None and coll is None:
    flask.redirect(flask.url_for('internal_error'))


# Our mock database -> replace with pymongo query to the mongodb database
# users = {'admin': {'password': generate_password_hash('robo@0'),
#                    'programs': ['all']},
#          'user': {'password': generate_password_hash('test'),
#                   'programs': ['4']}
#          }
# users = {'admin': {'password': 'pbkdf2:sha1:1000$tytMHt8x$121b8c130d98997228c100b13aa82acc9696c172'}}

''' serve additional static data (preview images, compressed source data)

When deploying, make sure WSGIScriptAlias is overridden by Apache's directive:

Alias "/data/" "/path/to/archive/data"
<Directory "/path/to/app/static/">
  Order allow,deny
  Allow from all
</Directory>

Check details at:
http://stackoverflow.com/questions/31298755/how-to-get-apache-to-serve-static-files-on-flask-webapp

Once deployed, comment the following definition:
'''


# FIXME:
@app.route('/data/<path:filename>')
def data_static(filename):
    return flask.send_from_directory(config.get('Path', 'path_to_website_data'), filename)


''' handle user login'''


class User(flask_login.UserMixin):
    pass


@login_manager.user_loader
def user_loader(username):
    # look up username in database:
    select = coll.find_one({'_id': username})
    if select is None:
        return

    user = User()
    user.id = username
    return user


@login_manager.request_loader
def request_loader(request):
    username = request.form.get('username')

    select = coll.find_one({'_id': username})
    if select is None:
        return

    user = User()
    user.id = username

    user.is_authenticated = check_password_hash(select['password'],
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
            return flask.render_template('template-login.html', fail=False)
    # print(flask.request.form['username'], flask.request.form['password'])

    username = flask.request.form['username']
    # check if username exists and passwords match
    select = coll.find_one({'_id': username})
    if select is not None and \
            check_password_hash(select['password'], flask.request.form['password']):
        user = User()
        user.id = username
        flask_login.login_user(user, remember=True)
        return flask.redirect(flask.url_for('root'))
    else:
        # serve template with flag fail=True to display fail message
        return flask.render_template('template-login.html', fail=True)


# serve root
@app.route('/')
@flask_login.login_required
def root():
    return flask.render_template('template-archive.html',
                                 user=flask_login.current_user.id,
                                 programs=['4', '41'],
                                 dates=['20160602', '20160726'])


# serve root
@app.route('/manage_users')
@flask_login.login_required
def manage_users():
    if flask_login.current_user.id == 'admin':
        # TODO: fetch users from the database:
        _users = {}
        cursor = coll.find()
        for usr in cursor:
            print(usr)
            _users[usr['_id']] = {'programs': [usr['programs']]}

        return flask.render_template('template-users.html',
                                     user=flask_login.current_user.id,
                                     users=_users)
    else:
        flask.abort(403)


@app.route('/add_user', methods=['GET'])
@flask_login.login_required
def add_user():
    try:
        user = flask.request.args['user']
        password = flask.request.args['password']
        programs = flask.request.args['programs']
        print(user, password, programs)
        result = coll.insert_one(
            {'_id': user,
             'password': generate_password_hash(password),
             'programs': programs}
        )
        # print(result.inserted_id)
        return 'success'
    except Exception as _e:
        print(_e)
        return _e


@app.route('/remove_user', methods=['GET', 'POST'])
@flask_login.login_required
def remove_user():
    return flask.redirect(flask.url_for('manage_users'))


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    flask_login.logout_user()
    return flask.redirect(flask.url_for('root'))


@app.errorhandler(500)
def internal_error(error):
    return '500 error'


@app.errorhandler(404)
def not_found(error):
    return '404 error'


@app.errorhandler(403)
def not_found(error):
    return '403 error: forbidden'


@login_manager.unauthorized_handler
def unauthorized_handler():
    return flask.redirect(flask.url_for('login'))


if __name__ == '__main__':
    app.run(host=config.get('Server', 'host'), port=config.get('Server', 'port'), threaded=True)
