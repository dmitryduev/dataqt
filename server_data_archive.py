"""
    Flask-based server for the Robo-AO Data Archive

    Dr Dmitry A. Duev @ Caltech, 2016
"""

from __future__ import print_function
from gevent import monkey
monkey.patch_all()

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
from urlparse import urlparse


def get_config(config_file='config.ini'):
    """
        load config data
    """
    try:
        abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
        _config = ConfigParser.RawConfigParser()
        _config.read(os.path.join(abs_path, config_file))
        # logger.debug('Successfully read in the config file {:s}'.format(args.config_file))

        ''' connect to mongodb database '''
        conf = dict()
        # paths:
        conf['path_archive'] = _config.get('Path', 'path_archive')
        # database access:
        conf['mongo_host'] = _config.get('Database', 'host')
        conf['mongo_port'] = int(_config.get('Database', 'port'))
        conf['mongo_db'] = _config.get('Database', 'db')
        conf['mongo_collection_obs'] = _config.get('Database', 'collection_obs')
        conf['mongo_collection_aux'] = _config.get('Database', 'collection_aux')
        conf['mongo_collection_pwd'] = _config.get('Database', 'collection_pwd')
        conf['mongo_collection_weather'] = _config.get('Database', 'collection_weather')
        conf['mongo_user'] = _config.get('Database', 'user')
        conf['mongo_pwd'] = _config.get('Database', 'pwd')

        ''' server location '''
        conf['server_host'] = _config.get('Server', 'host')
        conf['server_port'] = _config.get('Server', 'port')

        return conf

    except Exception as _e:
        print(_e)
        raise Exception('Failed to read in the config file')


def connect_to_db(_config):
    """ Connect to the mongodb database

    :return:
    """
    try:
        _client = MongoClient(host=_config['mongo_host'], port=_config['mongo_port'])
        _db = _client[_config['mongo_db']]
    except Exception as _e:
        print(_e)
        _db = None
    try:
        _db.authenticate(_config['mongo_user'], _config['mongo_pwd'])
    except Exception as _e:
        print(_e)
        _db = None
    try:
        _coll = _db[_config['mongo_collection_obs']]
    except Exception as _e:
        print(_e)
        _coll = None
    try:
        _coll_aux = _db[_config['mongo_collection_aux']]
    except Exception as _e:
        print(_e)
        _coll_aux = None
    try:
        _coll_weather = _db[_config['mongo_collection_weather']]
    except Exception as _e:
        print(_e)
        _coll_weather = None
    try:
        _coll_usr = _db[_config['mongo_collection_pwd']]
    except Exception as _e:
        print(_e)
        _coll_usr = None
    try:
        # build dictionary program num -> pi username
        cursor = _coll_usr.find()
        _program_pi = {}
        for doc in cursor:
            # handle admin separately
            if doc['_id'] == 'admin':
                continue
            _progs = doc['programs']
            for v in _progs:
                _program_pi[str(v)] = doc['_id'].encode('ascii', 'ignore')
                # print(program_pi)
    except Exception as _e:
        print(_e)
        _program_pi = None

    return _client, _db, _coll, _coll_usr, _coll_aux, _coll_weather, _program_pi


# import functools
# def background(f):
#     @functools.wraps(f)
#     def wrapper(*args, **kwargs):
#         jobid = uuid4().hex
#         key = 'job-{0}'.format(jobid)
#         skey = 'job-{0}-status'.format(jobid)
#         expire_time = 3600
#         redis.set(skey, 202)
#         redis.expire(skey, expire_time)
#
#         @copy_current_request_context
#         def task():
#             try:
#                 data = f(*args, **kwargs)
#             except:
#                 redis.set(skey, 500)
#             else:
#                 redis.set(skey, 200)
#                 redis.set(key, data)
#                 redis.expire(key, expire_time)
#             redis.expire(skey, expire_time)
#
#         gevent.spawn(task)
#         return jsonify({"job": jobid})
#     return wrapper

''' initialize the Flask app '''
app = flask.Flask(__name__)
app.secret_key = 'roboaokicksass'


def get_db(_config):
    """Opens a new database connection if there is none yet for the
    current application context.
    """
    if not hasattr(flask.g, 'client'):
        flask.g.client, flask.g.db, flask.g.coll, flask.g.coll_usr, \
        flask.g.coll_aux, flask.g.coll_weather, flask.g.program_pi = connect_to_db(_config)
    return flask.g.client, flask.g.db, flask.g.coll, flask.g.coll_usr, \
                flask.g.coll_aux, flask.g.coll_weather, flask.g.program_pi


@app.teardown_appcontext
def close_db(error):
    """Closes the database again at the end of the request."""
    if hasattr(flask.g, 'client'):
        flask.g.client.close()


login_manager = flask_login.LoginManager()

login_manager.init_app(app)

''' get config data '''
config = get_config(config_file='config.ini')
# print(config)

''' Connect to the mongodb database '''
# client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = connect_to_db(config)
# client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)

''' throw error 500 if could not connect to database '''
# if db is None and coll is None:
#     flask.redirect(flask.url_for('internal_error'))


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
    """
        Get files from the archive
    :param filename:
    :return:
    """
    _p, _f = os.path.split(filename)
    return flask.send_from_directory(os.path.join(config['path_archive'], _p), _f)


''' handle user login'''


class User(flask_login.UserMixin):
    pass


@login_manager.user_loader
def user_loader(username):
    # look up username in database:
    client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)
    select = coll_usr.find_one({'_id': username})
    if select is None:
        return

    user = User()
    user.id = username
    return user


@login_manager.request_loader
def request_loader(request):
    username = request.form.get('username')
    client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)
    # look up in the database
    select = coll_usr.find_one({'_id': username})
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
    # look up in the database first:
    client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)
    select = coll_usr.find_one({'_id': username})
    if select is not None and \
            check_password_hash(select['password'], flask.request.form['password']):
        user = User()
        user.id = username
        flask_login.login_user(user, remember=True)
        return flask.redirect(flask.url_for('root'))
    else:
        # serve template with flag fail=True to display fail message
        return flask.render_template('template-login.html', fail=True)


def stream_template(template_name, **context):
    """
        see: http://flask.pocoo.org/docs/0.11/patterns/streaming/
    :param template_name:
    :param context:
    :return:
    """
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.enable_buffering(5)
    return rv


def get_dates(user_id, coll, start=None, show_date=None):
    if start is None:
        # this is ~when we moved to KP:
        # start = datetime.datetime(2015, 10, 1)
        # by default -- last 30 days:
        start = datetime.datetime.utcnow() - datetime.timedelta(days=30)
    else:
        try:
            start = datetime.datetime.strptime(start, '%Y%m%d')
        except Exception as _e:
            print(_e)
            start = datetime.datetime.utcnow() - datetime.timedelta(days=30)
    if show_date is not None:
        try:
            show_date = datetime.datetime.strptime(show_date, '%Y%m%d')
        except Exception as _e:
            print(_e)
            show_date = None

    # dictionary: {date: {program_N: [observations]}}
    dates = dict()
    # programs = []
    if user_id == 'admin':
        # get everything;
        if show_date is None:
            cursor = coll.find({'date_utc': {'$gte': start}})
        else:
            cursor = coll.find({'date_utc': {'$gte': show_date,
                                             '$lt': show_date + datetime.timedelta(days=1)}})
    else:
        # get only programs accessible to this user marked as distributed:
        if show_date is None:
            cursor = coll.find({'date_utc': {'$gte': start},
                                'science_program.program_PI': user_id,
                                'distributed.status': True})
        else:
            cursor = coll.find({'date_utc': {'$gte': show_date,
                                             '$lt': show_date + datetime.timedelta(days=1)},
                                'science_program.program_PI': user_id,
                                'distributed.status': True})

    # iterate over query result:
    for obs in cursor:
        date = obs['date_utc'].strftime('%Y%m%d')
        # add key to dict if it is not there already:
        if date not in dates:
            dates[date] = dict()
        # add key for program if it is not there yet
        program_id = obs['science_program']['program_id']
        if program_id not in dates[date]:
            dates[date][program_id] = []
        dates[date][program_id].append(obs)

    # print(dates)
    # latest obs - first
    # dates = sorted(list(set(dates)), reverse=True)

    return dates


@app.route('/get_data', methods=['GET'])
@flask_login.login_required
def wget_script():
    url = urlparse(flask.request.url).netloc
    _date_str = flask.request.args['date']
    _date = datetime.datetime.strptime(_date_str, '%Y%m%d')
    _program = flask.request.args['program']

    user_id = flask_login.current_user.id
    # get db connection
    client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)

    # trying to get something you're not supposed to get?
    if user_id != 'admin' and program_pi[_program] != user_id:
        flask.abort(403)
    else:
        cursor = coll.find({'date_utc': {'$gte': _date,
                                         '$lt': _date + datetime.timedelta(days=1)},
                            'science_program.program_id': _program,
                            'distributed.status': True})
        response_text = '#!/usr/bin/env bash\n'
        for obs in cursor:
            response_text += 'wget http://{:s}/data/{:s}/{:s}/{:s}.tar.bz2\n'.format(url, _date_str,
                                                                                obs['_id'], obs['_id'])
        # print(response_text)

        # generate .sh file on the fly
        response = flask.make_response(response_text)
        response.headers['Content-Disposition'] = \
            'attachment; filename=program_{:s}_{:s}.wget.sh'.format(_program, _date_str)
        return response


# serve root
@app.route('/', methods=['GET'])
@flask_login.login_required
def root():
    if 'start' in flask.request.args and 'show_date' in flask.request.args:
        flask.abort(500)
    if 'show_date' in flask.request.args:
        show_date = flask.request.args['show_date']
    else:
        show_date = None
    if 'start' in flask.request.args:
        start = flask.request.args['start']
    else:
        start = None

    user_id = flask_login.current_user.id

    def iter_dates(_dates):
        """
            instead of first loading and then sending everything to user all at once,
             yield data for a single date at a time and stream to user
        :param _dates:
        :return:
        """
        if len(_dates) > 0:
            for _date in _dates:
                # print(_date, _dates[_date])
                yield _date, _dates[_date]
        else:
            yield None, None

    # get db connection
    client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)

    # get all dates:
    dates = get_dates(user_id, coll, start=start, show_date=show_date)

    return flask.Response(stream_template('template-archive.html',
                                          user=user_id,
                                          dates=iter_dates(dates)))
    # return flask.render_template('template-archive.html',
    #                              user=flask_login.current_user.id,
    #                              programs=['4', '41'],
    #                              dates=['20160602', '20160726'])


# manage users
@app.route('/manage_users')
@flask_login.login_required
def manage_users():
    if flask_login.current_user.id == 'admin':
        # fetch users from the database:
        _users = {}
        client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)
        cursor = coll_usr.find()
        for usr in cursor:
            # print(usr)
            if usr['programs'] == 'all':
                _users[usr['_id']] = {'programs': ['all']}
            else:
                _users[usr['_id']] = {'programs': [p.encode('ascii', 'ignore')
                                                   for p in usr['programs']]}

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
        programs = [p.strip().encode('ascii', 'ignore')
                    for p in flask.request.args['programs'].split(',')]
        # print(user, password, programs)
        # print(len(user), len(password), len(programs))
        if len(user) == 0 or len(password) == 0:
            return 'everything must be set'
        client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)
        result = coll_usr.insert_one(
            {'_id': user,
             'password': generate_password_hash(password),
             'programs': programs,
             'last_modified': datetime.datetime.now()}
        )
        # print(result.inserted_id)
        return 'success'
    except Exception as _e:
        print(_e)
        return _e


@app.route('/edit_user', methods=['GET'])
@flask_login.login_required
def edit_user():
    try:
        # print(flask.request.args)
        id = flask.request.args['_user']
        if id == 'admin':
            return 'Cannot remove the admin!'
        user = flask.request.args['edit-user']
        password = flask.request.args['edit-password']
        programs = [p.strip().encode('ascii', 'ignore')
                    for p in flask.request.args['edit-programs'].split(',')]
        # print(user, password, programs, id)
        # print(len(user), len(password), len(programs))
        if len(user) == 0:
            return 'username must be set'
        # keep old password:
        if len(password) == 0:
            client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)
            result = coll_usr.update_one(
                {'_id': id},
                {
                    '$set': {
                        '_id': user,
                        'programs': programs
                    },
                    '$currentDate': {'last_modified': True}
                }
            )
        # else change password too:
        else:
            client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)
            result = coll_usr.update_one(
                {'_id': id},
                {
                    '$set': {
                        '_id': user,
                        'password': generate_password_hash(password),
                        'programs': programs
                    },
                    '$currentDate': {'last_modified': True}
                }
            )
        # print(result.inserted_id)
        return 'success'
    except Exception as _e:
        print(_e)
        return _e


@app.route('/remove_user', methods=['GET', 'POST'])
@flask_login.login_required
def remove_user():
    try:
        # print(flask.request.args)
        # get username from request
        user = flask.request.args['user']
        if user == 'admin':
            return 'Cannot remove the admin!'
        # print(user)
        # try to remove
        client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = get_db(config)
        result = coll_usr.delete_one({'_id': user})
        return 'success'
    except Exception as _e:
        print(_e)
        return _e


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
    app.run(host=config['server_host'], port=config['server_port'], threaded=True)
