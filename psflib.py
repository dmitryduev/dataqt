from __future__ import print_function
import os
from pymongo import MongoClient
import inspect
import ConfigParser
import datetime
import pytz
import calendar
import argparse
import pyprind
import numpy as np
from astropy.io import fits
import traceback
import ast
import re
import vip
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker, TextArea
from matplotlib.patches import Rectangle
# import matplotlib.mlab as mlab
# import seaborn as sns
#
# # set up plotting
# sns.set_style('whitegrid')
# plt.close('all')
# sns.set_context('talk')


# Scale bars
class AnchoredSizeBar(AnchoredOffsetbox):
    def __init__(self, transform, size, label, loc,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, frameon=True):
        """
        Draw a horizontal bar with the size in data coordinate of the give axes.
        A label will be drawn underneath (center-aligned).

        pad, borderpad in fraction of the legend font size (or prop)
        sep in points.
        loc:
            'upper right'  : 1,
            'upper left'   : 2,
            'lower left'   : 3,
            'lower right'  : 4,
            'right'        : 5,
            'center left'  : 6,
            'center right' : 7,
            'lower center' : 8,
            'upper center' : 9,
            'center'       : 10
        """
        self.size_bar = AuxTransformBox(transform)
        self.size_bar.add_artist(Rectangle((0, 0), size, 0, fc='none', color='white', lw=3))

        self.txt_label = TextArea(label, dict(color='white', size='x-large', weight='normal'),
                                  minimumdescent=False)

        self._box = VPacker(children=[self.size_bar, self.txt_label],
                            align="center",
                            pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=self._box,
                                   prop=prop,
                                   frameon=frameon)


def get_xy_from_pipeline_settings_txt(_path, _first=True):
    """
        Get centroid position for a lucky-pipelined image
    :param _path:
    :param _first: output the x,y from the first run? if False, output from the last
    :return:
    """
    with open(os.path.join(_path, 'pipeline_settings.txt'), 'r') as _f:
        f_lines = _f.readlines()

    for l in f_lines:
        _tmp = re.search(r'\d\s+\d\s+\((\d+),(\d+)\),\((\d+),(\d+)\)\n', l)
        if _tmp is not None:
            _x = (int(_tmp.group(1)) + int(_tmp.group(3)))/2
            _y = (int(_tmp.group(2)) + int(_tmp.group(4)))/2
            if _first:
                break

    return _x, _y


def utc_now():
    return datetime.datetime.now(pytz.utc)


def get_config(_abs_path=None, _config_file='config.ini'):
    """ Get config data

    :return:
    """
    config = ConfigParser.RawConfigParser()

    if _config_file[0] not in ('/', '~'):
        if os.path.isfile(os.path.join(_abs_path, _config_file)):
            config.read(os.path.join(_abs_path, _config_file))
            if len(config.read(os.path.join(_abs_path, _config_file))) == 0:
                raise Exception('Failed to load config file')
        else:
            raise IOError('Failed to find config file')
    else:
        if os.path.isfile(_config_file):
            config.read(_config_file)
            if len(config.read(_config_file)) == 0:
                raise Exception('Failed to load config file')
        else:
            raise IOError('Failed to find config file')

    _config = dict()
    # planetary program number (do no crop planetary images!)
    _config['program_num_planets'] = int(config.get('Programs', 'planets'))
    # path to raw data:
    _config['path_raw'] = config.get('Path', 'path_raw')
    # path to lucky-pipeline data:
    _config['path_pipe'] = config.get('Path', 'path_pipe')
    # path to data archive:
    _config['path_archive'] = config.get('Path', 'path_archive')
    # path to temporary stuff:
    _config['path_tmp'] = config.get('Path', 'path_tmp')

    # telescope data (voor, o.a., Strehl computation)
    _tmp = ast.literal_eval(config.get('Strehl', 'Strehl_factor'))
    _config['telescope_data'] = dict()
    for telescope in 'Palomar', 'KittPeak':
        _config['telescope_data'][telescope] = ast.literal_eval(config.get('Strehl', telescope))
        _config['telescope_data'][telescope]['Strehl_factor'] = _tmp[telescope]

    # metrics [arcsec] to judge if an observation is good/bad
    _config['core_min'] = float(config.get('Strehl', 'core_min'))
    _config['halo_max'] = float(config.get('Strehl', 'halo_max'))

    # faint pipeline:
    _config['faint'] = dict()
    _config['faint']['win'] = int(config.get('Faint', 'win'))
    _config['faint']['n_threads'] = int(config.get('Faint', 'n_threads'))

    # pca pipeline
    _config['pca'] = dict()
    # path to PSF library:
    _config['pca']['path_psf_reference_library'] = config.get('Path', 'path_psf_reference_library')
    _config['pca']['path_psf_reference_library_short_names'] = \
        config.get('Path', 'path_psf_reference_library_short_names')
    # have to do it inside job_pca - otherwise will have to send hundreds of Mb
    # back and forth between redis queue and task consumer. luckily, it's pretty fast to do
    # with fits.open(path_psf_reference_library) as _lib:
    #     _config['pca']['psf_reference_library'] = _lib[0].data
    # _config['pca']['psf_reference_library_short_names'] = np.genfromtxt(path_psf_reference_library_short_names,
    #                                                              dtype='|S')

    _config['pca']['win'] = int(config.get('PCA', 'win'))
    _config['pca']['sigma'] = float(config.get('PCA', 'sigma'))
    _config['pca']['nrefs'] = float(config.get('PCA', 'nrefs'))
    _config['pca']['klip'] = float(config.get('PCA', 'klip'))

    _config['planets_prog_num'] = str(config.get('Programs', 'planets'))

    # seeing
    _config['seeing'] = dict()
    _config['seeing']['fit_model'] = config.get('Seeing', 'fit_model')
    _config['seeing']['win'] = int(config.get('Seeing', 'win'))

    # database access:
    _config['mongo_host'] = config.get('Database', 'host')
    _config['mongo_port'] = int(config.get('Database', 'port'))
    _config['mongo_db'] = config.get('Database', 'db')
    _config['mongo_collection_obs'] = config.get('Database', 'collection_obs')
    _config['mongo_collection_aux'] = config.get('Database', 'collection_aux')
    _config['mongo_collection_pwd'] = config.get('Database', 'collection_pwd')
    _config['mongo_collection_weather'] = config.get('Database', 'collection_weather')
    _config['mongo_user'] = config.get('Database', 'user')
    _config['mongo_pwd'] = config.get('Database', 'pwd')

    # server ip addresses
    _config['analysis_machine_external_host'] = config.get('Server', 'analysis_machine_external_host')
    _config['analysis_machine_external_port'] = config.get('Server', 'analysis_machine_external_port')

    # consider data from:
    _config['archiving_start_date'] = datetime.datetime.strptime(
        config.get('Misc', 'archiving_start_date'), '%Y/%m/%d')
    # how many times to try to rerun pipelines:
    _config['max_pipelining_retries'] = int(config.get('Misc', 'max_pipelining_retries'))

    # nap time -- do not interfere with the nightly operations On/Off
    _config['nap_time'] = dict()
    _config['nap_time']['sleep_at_night'] = eval(config.get('Misc', 'nap_time'))
    # local or UTC?
    _config['nap_time']['frame'] = config.get('Misc', 'nap_time_frame')
    _config['nap_time']['start'] = config.get('Misc', 'nap_time_start')
    _config['nap_time']['stop'] = config.get('Misc', 'nap_time_stop')

    _config['loop_interval'] = float(config.get('Misc', 'loop_interval'))

    return _config


def connect_to_db(_config):
    """ Connect to the mongodb database

    :return:
    """
    try:
        _client = MongoClient(host=_config['mongo_host'], port=_config['mongo_port'])
        _db = _client[_config['mongo_db']]
        # print('ok')
    except Exception as _e:
        print(_e)
        _db = None
    try:
        _db.authenticate(_config['mongo_user'], _config['mongo_pwd'])
        # print('auth ok')
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


def remove_from_lib(_psf_library_fits, _obs):
    try:
        with fits.open(_psf_library_fits) as hdulist:
            if len(hdulist) < 3:
                raise Exception('Kennot remove {:s}! Must have at least five PSF in the library'.format(_obs))

            # get index of the frame to be removed:
            index_obs = np.argmax(hdulist[-1].data['obs_names'] == _obs)

            # remove from table with names:
            hdulist[-1].remove_row(index_obs)
            # remove from images:
            hdulist.pop(index_obs)
            # update library:
            hdulist.writeto(_psf_library_fits, overwrite=True)
    except Exception as _e:
        traceback.print_exc()
        print(_e)
        raise Exception('Failed to remove {:s} from PSF library'.format(_obs))


def add_to_lib(_psf_library_fits, _path, _obs, _obj_name='unknown'):
    # last HDU contains a table with obs names and short names, the rest are actual PSFs
    try:
        # get frame:
        with fits.open(_path) as f:
            frame = f[0].data

        if not os.path.exists(_psf_library_fits):
            # library does not exist yet?
            frame_names = np.array([_obs])
            frame_short_names = np.array([_obj_name])
            # create a binary table to keep full obs names and "short" (object) names
            tbhdu = fits.BinTableHDU.from_columns([fits.Column(name='obs_names', format='80A',
                                                               array=frame_names),
                                                   fits.Column(name='obj_names', format='80A',
                                                               array=frame_short_names)])
            thdulist = fits.HDUList([fits.PrimaryHDU(frame), tbhdu])
            thdulist.writeto(_psf_library_fits, overwrite=True)
        else:
            # get library:
            with fits.open(_psf_library_fits) as hdulist:
                # append names:
                tbhdu = fits.BinTableHDU.from_columns(hdulist[-1].columns, nrows=hdulist[-1].data.shape[0]+1)
                tbhdu.data['obs_names'][-1] = _obs
                tbhdu.data['obj_names'][-1] = _obj_name
                hdulist[-1] = tbhdu
                # append frame:
                hdulist.insert(len(hdulist)-1, fits.ImageHDU(frame))
                # update library:
                hdulist.writeto(_psf_library_fits, overwrite=True)

    except:
        traceback.print_exc()
        raise Exception('Failed to add {:s} to PSF library'.format(_obs))


def in_fits(_psf_library_fits, _obs):
    """
        Check if _obs is in PSF library
    :param _psf_library_fits:
    :param _obs:
    :return:
    """
    try:
        with fits.open(_psf_library_fits) as hdulist:
            if _obs in hdulist[-1].data['obs_names']:
                return True
            else:
                return False
    except Exception as _e:
        traceback.print_exc()
        print(_e)
        return False


if __name__ == '__main__':
    ''' Create command line argument parser '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Manage Robo-AO')

    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)

    args = parser.parse_args()
    config_file = args.config_file

    ''' script absolute location '''
    abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))

    ''' load config data '''
    try:
        config = get_config(_abs_path=abs_path, _config_file=args.config_file)
    except IOError:
        config = get_config(_abs_path=abs_path, _config_file='config.ini')

    try:
        # connect to db, pull out collections:
        client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = connect_to_db(config)

        # PSF library fits file name
        psf_library_fits = os.path.join(config['path_archive'], 'psf_library.fits')

        ''' get aux data '''
        print(config['archiving_start_date'])
        select_aux = coll_aux.find({'_id': {'$gte': config['archiving_start_date'].strftime('%Y%m%d')}})

        if select_aux.count() > 0:

            # iterate over all dates from coll_aux:
            for ob_aux in select_aux:

                date_str = ob_aux['_id']
                date = datetime.datetime.strptime(ob_aux['_id'], '%Y%m%d')

                print(date)

                # path to store data for individual frames:
                _path_out = os.path.join(config['path_archive'], date_str, 'summary', 'psflib')

                ''' TODO: create psflib field in aux collection in the database, or make sure it's there '''
                # check when last updated
                if 'psf_lib' not in ob_aux:
                    # init if necessary
                    coll_aux.update_one(
                        {'_id': date_str},
                        {
                            '$set': {
                                'psf_lib': {}
                            }
                        }
                    )
                    # reload:
                    ob_aux = coll_aux.find_one({'_id': date_str})
                    # frames: {'obs_name': {'in_library': False, 'outdated': False,
                    #                       'failed': False, 'last_modified': datetime}

                ''' grab obs credentials from the database '''
                # dict to store query to be executed on the main collection (with obs data):
                query = dict()

                # query['filter'] = 'Si'

                # query['seeing.median'] = {'$ne': None}

                query['date_utc'] = {'$gte': date, '$lt': date + datetime.timedelta(days=1)}

                # exclude planetary data:
                query['science_program.program_id'] = {'$ne': '24'}

                # azimuth and elevation range:
                # query['coordinates.azel.0'] = {'$gte': 0 * np.pi / 180.0, '$lte': 360 * np.pi / 180.0}
                # query['coordinates.azel.1'] = {'$gte': 0 * np.pi / 180.0, '$lte': 90 * np.pi / 180.0}

                # consider reliable Strehls only:
                # query['pipelined.automated.strehl.flag'] = {'$eq': 'OK'}

                # consider only 'done' stuff:
                query['pipelined.automated.status.done'] = True

                # discard observations marked as "zero_flux" by the automated pipeline
                query['pipelined.automated.classified_as'] = {'$nin': ['zero_flux', 'failed']}

                # execute query:
                if len(query) > 0:
                    # print('executing query:\n{:s}'.format(query))
                    select = coll.find(query)

                    if select.count() > 0:

                        for ob in select:

                            try:

                                # see lucidchart.com for the processing flowchart

                                ob_id = ob['_id']
                                # field names in MongoDB cannot contain dots:
                                ob_id_db = ob['_id'].split('.')[0]

                                in_db = ob_id_db in ob_aux['psf_lib']

                                # last pipelined:
                                last_pipelined = ob['pipelined']['automated']['last_modified']

                                execute_processing = False

                                ''' in DB? '''
                                if in_db:
                                    ''' Done? '''
                                    done = ob_aux['psf_lib'][ob_id_db]['done']
                                    if done:
                                        ''' Updated? '''
                                        updated = np.abs(last_pipelined -
                                                    ob_aux['psf_lib'][ob_id_db]['last_modified']).total_seconds() > 60
                                        if updated:
                                            # mark in DB:
                                            coll_aux.update_one(
                                                {'_id': date_str},
                                                {
                                                    '$set': {
                                                        'psf_lib.{:s}.done'.format(ob_id_db): False,
                                                        'psf_lib.{:s}.updated'.format(ob_id_db): True,
                                                        'psf_lib.{:s}.last_modified'.format(ob_id_db): last_pipelined
                                                    }
                                                }
                                            )
                                            #
                                            execute_processing = True

                                        ''' Check status from web interface '''
                                        enqueued = ob_aux['psf_lib'][ob_id_db]['enqueued']
                                        if enqueued:
                                            status = ob_aux['psf_lib'][ob_id_db]['status']
                                            if status == 'add_to_lib':
                                                try:
                                                    in_lib = ob_aux['psf_lib'][ob_id_db]['in_lib']
                                                    if in_lib or updated:
                                                        # remove from lib, mark updated: false
                                                        remove_from_lib(_psf_library_fits=psf_library_fits,
                                                                        _obs=ob_id_db)
                                                        # execute add to psf lib
                                                        add_to_lib(_psf_library_fits=psf_library_fits,
                                                                   _path=os.path.join(_path_out,
                                                                                      '{:s}.fits'.format(ob_id)),
                                                                   _obs=ob_id_db, _obj_name=ob['name'])
                                                    else:
                                                        # execute add to psf lib
                                                        add_to_lib(_psf_library_fits=psf_library_fits,
                                                                   _path=os.path.join(_path_out,
                                                                                      '{:s}.fits'.format(ob_id)),
                                                                   _obs=ob_id_db, _obj_name=ob['name'])

                                                    coll_aux.update_one(
                                                        {'_id': date_str},
                                                        {
                                                            '$set': {
                                                                'psf_lib.{:s}.in_lib'.format(ob_id_db): True
                                                            }
                                                        }
                                                    )
                                                except Exception as e:
                                                    print(e)
                                                    coll_aux.update_one(
                                                        {'_id': date_str},
                                                        {
                                                            '$set': {
                                                                'psf_lib.{:s}.in_lib'.format(ob_id_db): False
                                                            }
                                                        }
                                                    )

                                            elif status == 'remove_from_lib':
                                                try:
                                                    # check if actually in lib
                                                    if in_fits(_psf_library_fits=psf_library_fits, _obs=ob_id_db):
                                                        # remove from lib
                                                        remove_from_lib(_psf_library_fits=psf_library_fits,
                                                                        _obs=ob_id_db)
                                                except Exception as e:
                                                    print(e)

                                                coll_aux.update_one(
                                                    {'_id': date_str},
                                                    {
                                                        '$set': {
                                                            'psf_lib.{:s}.in_lib'.format(ob_id_db): False
                                                        }
                                                    }
                                                )

                                            coll_aux.update_one(
                                                {'_id': date_str},
                                                {
                                                    '$set': {
                                                        'psf_lib.{:s}.updated'.format(ob_id_db): False,
                                                        'psf_lib.{:s}.enqueued'.format(ob_id_db): False,
                                                        'psf_lib.{:s}.status'.format(ob_id_db): None
                                                    }
                                                }
                                            )

                                    else:
                                        ''' Process if tried not too many times and high_flux or faint '''
                                        tried_too_many_times = ob_aux['psf_lib'][ob_id_db]['retries'] \
                                                                > config['max_pipelining_retries']
                                        tag = str(ob['pipelined']['automated']['classified_as'])
                                        if (not tried_too_many_times) and (tag not in ('failed', 'zero_flux')):
                                            execute_processing = True

                                else:
                                    # init entry for a new record:
                                    coll_aux.update_one(
                                        {'_id': date_str},
                                        {
                                            '$set': {
                                                'psf_lib.{:s}'.format(ob_id_db): {
                                                    'done': False,
                                                    'in_lib': False,
                                                    'updated': False,
                                                    'enqueued': False,
                                                    'status': False,
                                                    'retries': 0,
                                                    'last_modified': utc_now()
                                                }
                                            }
                                        }
                                    )
                                    #
                                    execute_processing = True

                                if execute_processing:

                                    # number of processing attempts++
                                    coll_aux.update_one(
                                        {'_id': date_str},
                                        {
                                            '$inc': {
                                                    'psf_lib.{:s}.retries'.format(ob_id_db): 1
                                                }
                                        }
                                    )

                                    ''' preprocess 100p.fits, high-pass, cut '''
                                    tag = str(ob['pipelined']['automated']['classified_as'])
                                    _path = os.path.join(config['path_pipe'], date_str, tag, ob_id)
                                    _fits_name = '100p.fits'

                                    if os.path.exists(os.path.join(_path, _fits_name)):

                                        print(os.path.join(_path, _fits_name))

                                        with fits.open(os.path.join(_path, _fits_name)) as _hdu:
                                            scidata = _hdu[0].data

                                        _win = 100
                                        y, x = get_xy_from_pipeline_settings_txt(_path)
                                        x *= 2.0
                                        y *= 2.0
                                        x, y = map(int, [x, y])

                                        # out of the frame? do not try to fix that, just skip!
                                        if x - _win < 0 or x + _win + 1 >= scidata.shape[0] \
                                                or y - _win < 0 or y + _win + 1 >= scidata.shape[1]:
                                            coll_aux.update_one(
                                                {'_id': date_str},
                                                {
                                                    '$set': {
                                                        'psf_lib.{:s}.done'.format(ob_id_db): False,
                                                        'psf_lib.{:s}.last_modified'.format(ob_id_db): utc_now()
                                                    }
                                                }
                                            )
                                            continue

                                        _trimmed_frame = scidata[x - _win: x + _win + 1,
                                                                 y - _win: y + _win + 1]

                                        # Filter the trimmed frame with IUWT filter, 2 coeffs
                                        filtered_frame = (vip.var.cube_filter_iuwt(
                                            np.reshape(_trimmed_frame,
                                                       (1, np.shape(_trimmed_frame)[0], np.shape(_trimmed_frame)[1])),
                                            coeff=5, rel_coeff=2))
                                        print(filtered_frame.shape)

                                        mean_y, mean_x, fwhm_y, fwhm_x, amplitude, theta = \
                                            (vip.var.fit_2dgaussian(filtered_frame[0], crop=True, cropsize=50, debug=False,
                                                                    full_output=True))
                                        _fwhm = np.mean([fwhm_y, fwhm_x])

                                        # Print the resolution element size
                                        # print('Using resolution element size = ', _fwhm)
                                        if _fwhm < 2:
                                            _fwhm = 2.0
                                            # print('Too small, changing to ', _fwhm)
                                        _fwhm = int(_fwhm)

                                        # Center the filtered frame
                                        centered_cube, shy, shx = \
                                            (vip.calib.cube_recenter_gauss2d_fit(array=filtered_frame, xy=(_win, _win),
                                                                                 fwhm=_fwhm,
                                                                                 subi_size=6, nproc=1, full_output=True))

                                        centered_frame = centered_cube[0]

                                        ''' create preview '''
                                        fig = plt.figure()
                                        ax = fig.add_subplot(111)

                                        ''' cropped image: '''
                                        # save cropped image
                                        plt.close('all')
                                        fig = plt.figure()
                                        fig.set_size_inches(3, 3, forward=False)
                                        # ax = fig.add_subplot(111)
                                        ax = plt.Axes(fig, [0., 0., 1., 1.])
                                        ax.set_axis_off()
                                        fig.add_axes(ax)
                                        ax.imshow(centered_frame, cmap=plt.cm.magma, origin='lower',
                                                  interpolation='nearest')

                                        # add scale bar:
                                        _drizzled = True
                                        _fow_x = 36
                                        _pix_x = 1024
                                        # draw a horizontal bar with length of 0.1*x_size
                                        # (ax.transData) with a label underneath.
                                        bar_len = centered_frame.shape[0] * 0.1
                                        mltplr = 2 if _drizzled else 1
                                        bar_len_str = '{:.1f}'.format(bar_len * _fow_x / _pix_x / mltplr)
                                        asb = AnchoredSizeBar(ax.transData,
                                                              bar_len,
                                                              bar_len_str[0] + r"$^{\prime\prime}\!\!\!.$" + bar_len_str[
                                                                  -1],
                                                              loc=4, pad=0.3, borderpad=0.5, sep=10, frameon=False)
                                        ax.add_artist(asb)

                                        # plt.show()

                                        ''' store both fits and png for the web interface '''
                                        png_name = '{:s}.png'.format(ob_id)
                                        fits_name = '{:s}.png'.format(ob_id)
                                        if not (os.path.exists(_path_out)):
                                            os.makedirs(_path_out)
                                        fig.savefig(os.path.join(_path_out, png_name), dpi=300)
                                        # save box around selected object:
                                        hdu = fits.PrimaryHDU(centered_frame)
                                        hdu.writeto(os.path.join(_path_out, '{:s}.fits'.format(ob_id)),
                                                    overwrite=True)

                                        # mark done:
                                        coll_aux.update_one(
                                            {'_id': date_str},
                                            {
                                                '$set': {
                                                    'psf_lib.{:s}.done'.format(ob_id_db): True,
                                                    'psf_lib.{:s}.last_modified'.format(ob_id_db): last_pipelined
                                                }
                                            }
                                        )

                                        # TODO: this is a button to add to the web interface
                                        ''' force_redo automated.pca for all obs on that date '''

                            except Exception as e:
                                traceback.print_exc()
                                print(e)
                                coll_aux.update_one(
                                    {'_id': date_str},
                                    {
                                        '$set': {
                                            'psf_lib.{:s}.done'.format(ob_id_db): False,
                                            'psf_lib.{:s}.last_modified'.format(ob_id_db): utc_now()
                                        }
                                    }
                                )

    except Exception as e:
        traceback.print_exc()
        print(e)

    finally:
        # plt.show()

        client.close()
