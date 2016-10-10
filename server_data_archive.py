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
from astropy.coordinates import SkyCoord
from astropy import units
import pyvo as vo
from PIL import Image
import urllib2
from cStringIO import StringIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np


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

        # VO server:
        conf['vo_server'] = _config.get('Misc', 'vo_url')

        return conf

    except Exception as _e:
        print(_e)
        raise Exception('Failed to read in the config file')


def parse_obs_name(_obs, _program_pi):
    """
        Parse Robo-AO observation name
    :param _obs:
    :param _program_pi: dict program_num -> PI
    :return:
    """
    # parse name:
    _tmp = _obs.split('_')
    # program num. it will be a string in the future
    _prog_num = str(_tmp[0])
    # who's pi?
    if _prog_num in _program_pi.keys():
        _prog_pi = _program_pi[_prog_num]
    else:
        # play safe if pi's unknown:
        _prog_pi = 'admin'
    # stack name together if necessary (if contains underscores):
    _sou_name = '_'.join(_tmp[1:-5])
    # code of the filter used:
    _filt = _tmp[-4:-3][0]
    # date and time of obs:
    _date_utc = datetime.datetime.strptime(_tmp[-2] + _tmp[-1], '%Y%m%d%H%M%S.%f')
    # camera:
    _camera = _tmp[-5:-4][0]
    # marker:
    _marker = _tmp[-3:-2][0]

    return _prog_num, _prog_pi, _sou_name, _filt, _date_utc, _camera, _marker


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


def get_dates(user_id, coll, start=None, stop=None):
    if start is None:
        # this is ~when we moved to KP:
        # start = datetime.datetime(2015, 10, 1)
        # by default -- last 30 days:
        start = datetime.datetime.utcnow() - datetime.timedelta(days=10)
    else:
        try:
            start = datetime.datetime.strptime(start, '%Y%m%d')
        except Exception as _e:
            print(_e)
            start = datetime.datetime.utcnow() - datetime.timedelta(days=10)

    if stop is None:
        stop = datetime.datetime.utcnow()
    else:
        try:
            stop = datetime.datetime.strptime(stop, '%Y%m%d')
        except Exception as _e:
            print(_e)
            stop = datetime.datetime.utcnow()

    # create index not to perform in-memory sorting:
    coll.create_index([('date_utc', -1)])

    # dictionary: {date: {program_N: [observations]}}
    dates = OrderedDict()
    # programs = []
    if user_id == 'admin':
        # get everything;
        cursor = coll.find({'date_utc': {'$gte': start, '$lt': stop}})
    else:
        # get only programs accessible to this user marked as distributed:
        cursor = coll.find({'date_utc': {'$gte': start, '$lt': stop},
                            'science_program.program_PI': user_id,
                            'distributed.status': True})

    # iterate over query result:
    try:
        for obs in cursor.sort([('date_utc', -1)]):
            date = obs['date_utc'].strftime('%Y%m%d')
            # add key to dict if it is not there already:
            if date not in dates:
                dates[date] = dict()
            # add key for program if it is not there yet
            program_id = obs['science_program']['program_id']
            if program_id not in dates[date]:
                dates[date][program_id] = []
            dates[date][program_id].append(obs)
    except Exception as _e:
        print(_e)

    # print(dates)
    # latest obs - first
    # dates = sorted(list(set(dates)), reverse=True)

    return dates


def get_aux(dates, coll_aux):
    """
        Get auxiliary data for a list of dates
    :param dates:
    :param coll_aux:
    :return:
    """

    aux = dict()

    for date in dates:
        aux[date] = OrderedDict()

        cursor = coll_aux.find({'_id': date})
        for date_data in cursor:
            for key in ('seeing', 'contrast_curve', 'strehl'):
                aux[date][key] = dict()
                aux[date][key]['done'] = True if (key in date_data and date_data[key]['done']) else False
                if key == 'seeing':
                    # for seeing data, fetch frame names to show in a 'movie'
                    aux[date][key]['frames'] = []
                    # sort by time, not by name:
                    ind_sort = np.argsort([frame[1] for frame in date_data[key]['frames']])
                    for frame in np.array(date_data[key]['frames'])[ind_sort]:
                        aux[date][key]['frames'].append(frame[0] + '.png')

    return aux


@app.route('/_get_fits_header')
@flask_login.login_required
def get_fits_header():
    """
        Get FITS header for a _source_ observed on a _date_
    :return: jsonified dictionary with the header / empty dict if failed
    """
    user_id = flask_login.current_user.id

    # get parameters from the AJAX GET request
    _obs = flask.request.args.get('source', 0, type=str)

    _, _, coll, _, _, _, _program_pi = get_db(config)

    # trying to steal stuff?
    _program, _, _, _, _, _, _ = parse_obs_name(_obs, _program_pi)
    if user_id != 'admin' and _program_pi[_program] != user_id:
        # flask.abort(403)
        return flask.jsonify(result={})

    cursor = coll.find({'_id': _obs})

    try:
        if cursor.count() == 1:
            for obs in cursor:
                header = obs['pipelined']['automated']['fits_header']
            return flask.jsonify(result=OrderedDict(header))
        # not found in the database?
        else:
            return flask.jsonify(result={})
    except Exception as _e:
        print(_e)
        return flask.jsonify(result={})


@app.route('/_get_vo_image', methods=['GET'])
@flask_login.login_required
def get_vo_image():
    user_id = flask_login.current_user.id

    # get parameters from the AJAX GET request
    _obs = flask.request.args.get('source', 0, type=str)

    _, _, coll, _, _, _, _program_pi = get_db(config)

    # trying to steal stuff?
    _program, _, _, _, _, _, _ = parse_obs_name(_obs, _program_pi)
    if user_id != 'admin' and _program_pi[_program] != user_id:
        # flask.abort(403)
        return flask.jsonify(result={})

    # print(_obs)
    cursor = coll.find({'_id': _obs})

    try:
        if cursor.count() == 1:
            for obs in cursor:
                header = obs['pipelined']['automated']['fits_header']
                if 'TELDEC' in header and 'TELRA' in header:
                    c = SkyCoord(header['TELRA'][0], header['TELDEC'][0],
                                 unit=(units.hourangle, units.deg), frame='icrs')
                    # print(c)
                    # print(c.ra.deg, c.dec.deg)

                    vo_url = config['vo_server']

                    survey_filter = {'dss2': 'r', '2mass': 'h'}
                    output = {}

                    for survey in survey_filter:

                        # TODO: use image size + scale from config. For now it's 36"x36" times 2 (0.02 deg^2)
                        previews = vo.imagesearch(vo_url, pos=(c.ra.deg, c.dec.deg),
                                                  size=(0.01*2, 0.01*2), format='image/png',
                                                  survey=survey)
                        if previews.nrecs == 0:
                            continue
                        # get url:
                        image_url = [image.getdataurl() for image in previews
                                     if image.title == survey + survey_filter[survey]][0]

                        _file = StringIO(urllib2.urlopen(image_url).read())
                        survey_image = np.array(Image.open(_file))

                        fig = plt.figure(figsize=(3, 3))
                        ax = fig.add_subplot(111)
                        ax.imshow(survey_image, origin='lower', interpolation='nearest', cmap=plt.cm.magma)
                        ax.grid('off')
                        ax.set_axis_off()
                        fig.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, right=1, left=0)
                        plt.margins(0, 0)
                        # flip x axis for it to look like our images:
                        plt.gca().invert_xaxis()
                        plt.gca().invert_yaxis()

                        canvas = FigureCanvas(fig)
                        png_output = StringIO()
                        canvas.print_png(png_output)
                        png_output = png_output.getvalue().encode('base64')

                        output['{:s}_{:s}'.format(survey, survey_filter[survey])] = png_output

            return flask.jsonify(result=output)
        # not found in the database?
        else:
            return flask.jsonify(result={})
    except Exception as _e:
        print(_e)
        return flask.jsonify(result={})


@app.route('/get_data', methods=['GET'])
@flask_login.login_required
def wget_script():
    """
        Generate bash script to fetch all data for date/program with wget
    :return:
    """
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

    if 'start' in flask.request.args:
        start = flask.request.args['start']
    else:
        start = None
    if 'stop' in flask.request.args:
        stop = flask.request.args['stop']
    else:
        stop = None

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
    dates = get_dates(user_id, coll, start=start, stop=stop)
    # get aux info:
    aux = get_aux(dates.keys(), coll_aux)

    return flask.Response(stream_template('template-archive.html',
                                          user=user_id,
                                          aux=aux,
                                          dates=iter_dates(dates)))


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
