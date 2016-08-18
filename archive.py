"""
    Data Archiving for Robo-AO

    Generates stuff to be displayed on the archive
    Updates the database

    Dmitry Duev (Caltech) 2016
"""
from __future__ import print_function

import ConfigParser
import argparse
import inspect
import os
import logging
import datetime
import pytz
import time
from astropy.io import fits
from pymongo import MongoClient
import sys
import re
from collections import OrderedDict
import numpy as np
from copy import deepcopy

from skimage import exposure, img_as_float
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker, TextArea
import matplotlib.pyplot as plt
import seaborn as sns

# import numba
from huey import RedisHuey
huey = RedisHuey('roboao.archive', result_store=True)

# set up plotting
sns.set_style('whitegrid')
# sns.set_palette(sns.diverging_palette(10, 220, sep=80, n=7))
plt.close('all')
sns.set_context('talk')


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


@huey.task()
# @numba.jit
def job_pca(_obs):
    tic = time.time()
    a = 0
    for i in range(100):
        for j in range(100):
            for k in range(1000):
                a += 3**2
    print('It took {:.2f} s to finish the job on {:s}'.format(time.time() - tic, _obs))
    _logger.debug('done a pca job on {:s}'.format(_obs))
    _coll.update_one(
        {'_id': _obs},
        {
            '$set': {
                'pipelined.pca.status.done': True,
            }
        }
    )
    return True


@huey.task()
# @numba.jit
def job_strehl(_obs):
    tic = time.time()
    a = 0
    for i in range(100):
        for j in range(100):
            for k in range(1000):
                a += 3**2
    print('It took {:.2f} s to finish the job on {:s}'.format(time.time() - tic, _obs))

    return True


def utc_now():
    return datetime.datetime.now(pytz.utc)


def naptime(nap_time_start, nap_time_stop):
    """
        Return time to sleep in seconds for the archiving engine
        before waking up to rerun itself.
         In the daytime, it's 1 hour
         In the nap time, it's nap_time_start_utc - utc_now()
    :return:
    """
    now_local = datetime.datetime.now()
    # TODO: finish!


def load_fits(fin):
    with fits.open(fin) as _f:
        scidata = _f[0].data
    return scidata


def scale_image(image, correction='local'):
    """

    :param image:
    :param correction: 'local', 'log', or 'global'
    :return:
    """
    # scale image for beautification:
    scidata = deepcopy(image)
    norm = np.max(np.max(scidata))
    mask = scidata <= 0
    scidata[mask] = 0
    scidata = np.uint16(scidata / norm * 65535)

    # add more contrast to the image:
    if correction == 'log':
        return exposure.adjust_log(img_as_float(scidata/norm) + 1, 1)
    elif correction == 'global':
        p_1, p_2 = np.percentile(scidata, (5, 100))
        return exposure.rescale_intensity(scidata, in_range=(p_1, p_2))
    elif correction == 'local':
        # perform local histogram equalization instead:
        return exposure.equalize_adapthist(scidata, clip_limit=0.03)
    else:
        raise Exception('Contrast correction option not recognized')


def generate_pca_images(_out_path, _sou_dir, _preview_img, _cc,
                        _fow_x=36, _pix_x=1024, _drizzle=True):
    """
            Generate preview images for the pca pipeline

        :param _out_path:
        :param _sou_dir:
        :param _preview_img:
        :param _cc:
        :param _fow_x: full FoW in arcseconds in the x direction
        :param _pix_x: original full frame size in pixels
        :param _drizzle: drizzle on or off?

        :return:
        """
    try:
        ''' plot psf-subtracted image '''
        plt.close('all')
        fig = plt.figure(_sou_dir)
        fig.set_size_inches(3, 3, forward=False)
        # ax = fig.add_subplot(111)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(_preview_img, cmap='gray', origin='lower', interpolation='nearest')
        # add scale bar:
        # draw a horizontal bar with length of 0.1*x_size
        # (ax.transData) with a label underneath.
        bar_len = _preview_img.shape[0] * 0.1
        bar_len_str = '{:.1f}'.format(bar_len * _fow_x / _pix_x / 2) if _drizzle \
            else '{:.1f}'.format(bar_len * _fow_x / _pix_x)
        asb = AnchoredSizeBar(ax.transData,
                              bar_len,
                              bar_len_str[0] + r"$^{\prime\prime}\!\!\!.$" + bar_len_str[-1],
                              loc=4, pad=0.3, borderpad=0.5, sep=10, frameon=False)
        ax.add_artist(asb)

        # save figure
        fig.savefig(os.path.join(_out_path, _sou_dir + '_pca.png'), dpi=300)

        ''' plot the contrast curve '''
        plt.close('all')
        fig = plt.figure('Contrast curve for {:s}'.format(_sou_dir), figsize=(8, 3.5), dpi=200)
        ax = fig.add_subplot(111)
        ax.set_title(_sou_dir)  # , fontsize=14)
        ax.plot(_cc[:, 0], -2.5 * np.log10(_cc[:, 1]), 'k-', linewidth=2.5)
        ax.set_xlim([0.2, 1.45])
        ax.set_xlabel('Separation [arcseconds]')  # , fontsize=18)
        ax.set_ylabel('Contrast [$\Delta$mag]')  # , fontsize=18)
        ax.set_ylim([0, 8])
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.grid(linewidth=0.5)
        plt.tight_layout()
        fig.savefig(os.path.join(_out_path, _sou_dir + '_contrast_curve.png'), dpi=200)
    except Exception as _e:
        print(_e)
        return False

    return True


def empty_db_record():
    time_now_utc = utc_now()
    return {
            '_id': None,
            'date_added': time_now_utc,
            'name': None,
            'alternative_names': [],
            'science_program': {
                'program_id': None,
                'program_PI': None,
                'distributed': None
            },
            'date_utc': None,
            'telescope': None,
            'camera': None,
            'filter': None,
            'exposure': None,
            'magnitude': None,
            'coordinates': {
                'epoch': None,
                'radec': None,
                'radec_str': None,
                'azel': None
            },
            'pipelined': {
                'automated': {
                    'status': {
                        'done': False,
                        'preview': False
                    },
                    'location': [],
                    'classified_as': None,
                    'fits_header': {},
                    'last_modified': time_now_utc
                },
                'faint': {
                    'status': {
                        'done': False,
                        'preview': False,
                        'retries': 0
                    },
                    'location': [],
                    'last_modified': time_now_utc
                },
                'pca': {
                    'status': {
                        'done': False,
                        'preview': False,
                        'retries': 0
                    },
                    'location': [],
                    'contrast_curve': None,
                    'last_modified': time_now_utc
                },
                'strehl': {
                    'status': {
                        'done': False,
                        'retries': 0
                    },
                    'ratio_percent': None,
                    'core_arcsec': None,
                    'halo_arcsec': None,
                    'fwhm_arcsec': None,
                    'flag': None,
                    'last_modified': time_now_utc
                }
            },

            'seeing': {
                'median': None,
                'mean': None,
                'last_modified': time_now_utc
            },
            'bzip2': {
                'location': [],
                'last_modified': time_now_utc
            },
            'raw_data': {
                'location': [],
                'data': [],
                'last_modified': time_now_utc
            },
            'comment': None
        }


def set_up_logging(_path='logs', _name='archive', _level=logging.DEBUG, _mode='w'):
    """ Set up logging

    :param _path:
    :param _name:
    :param _level: DEBUG, INFO, etc.
    :param _mode: overwrite log-file or append: w or a
    :return: logger instance
    """

    if not os.path.exists(_path):
        os.makedirs(_path)
    utc_now = datetime.datetime.utcnow()

    # http://www.blog.pythonlibrary.org/2012/08/02/python-101-an-intro-to-logging/
    _logger = logging.getLogger(_name)

    _logger.setLevel(_level)
    # create the logging file handler
    fh = logging.FileHandler(os.path.join(_path,
                                          '{:s}.{:s}.log'.format(_name, utc_now.strftime('%Y%m%d'))),
                             mode=_mode)
    logging.Formatter.converter = time.gmtime

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)

    # add handler to logger object
    _logger.addHandler(fh)

    return _logger


def get_config(_config_file='config.ini'):
    """ Get config data

    :return:
    """
    config = ConfigParser.RawConfigParser()

    if _config_file[0] not in ('/', '~'):
        if os.path.isfile(os.path.join(abs_path, _config_file)):
            config.read(os.path.join(abs_path, _config_file))
            if len(config.read(os.path.join(abs_path, _config_file))) == 0:
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
    # path to (standard) pipeline data:
    _config['path_pipe'] = config.get('Path', 'path_pipe')
    # path to Becky-pipeline data:
    _config['path_pca'] = config.get('Path', 'path_pca')
    # path to seeing plots:
    _config['path_seeing'] = config.get('Path', 'path_seeing')
    # website data dwelling place:
    _config['path_to_website_data'] = config.get('Path', 'path_to_website_data')

    # database access:
    _config['mongo_host'] = config.get('Database', 'host')
    _config['mongo_port'] = int(config.get('Database', 'port'))
    _config['mongo_db'] = config.get('Database', 'db')
    _config['mongo_collection_obs'] = config.get('Database', 'collection_obs')
    _config['mongo_collection_pwd'] = config.get('Database', 'collection_pwd')
    _config['mongo_user'] = config.get('Database', 'user')
    _config['mongo_pwd'] = config.get('Database', 'pwd')

    # server ip addresses
    _config['analysis_machine_external_host'] = config.get('Server', 'analysis_machine_external_host')
    _config['analysis_machine_external_port'] = config.get('Server', 'analysis_machine_external_port')

    # consider data from:
    _config['archiving_start_date'] = datetime.datetime.strptime(
                            config.get('Auxiliary', 'archiving_start_date'), '%Y/%m/%d')

    return _config


def connect_to_db(_config, _logger):
    """ Connect to the mongodb database

    :return:
    """
    try:
        client = MongoClient(host=_config['mongo_host'], port=_config['mongo_port'])
        _db = client[_config['mongo_db']]
        _logger.debug('Successfully connected to the Robo-AO database at {:s}:{:d}'.
                      format(_config['mongo_host'], _config['mongo_port']))
    except Exception as _e:
        _db = None
        _logger.error(_e)
        _logger.error('Failed to connect to the Robo-AO database at {:s}:{:d}'.
                      format(_config['mongo_host'], _config['mongo_port']))
    try:
        _db.authenticate(_config['mongo_user'], _config['mongo_pwd'])
        _logger.debug('Successfully authenticated with the Robo-AO database at {:s}:{:d}'.
                      format(_config['mongo_host'], _config['mongo_port']))
    except Exception as _e:
        _db = None
        _logger.error(_e)
        _logger.error('Authentication failed for the Robo-AO database at {:s}:{:d}'.
                      format(_config['mongo_host'], _config['mongo_port']))
    try:
        _coll = _db[_config['mongo_collection_obs']]
        # cursor = coll.find()
        # for doc in cursor:
        #     print(doc)
        _logger.debug('Using collection {:s} with obs data in the database'.
                      format(_config['mongo_collection_obs']))
    except Exception as _e:
        _coll = None
        _logger.error(_e)
        _logger.error('Failed to use a collection {:s} with obs data in the database'.
                      format(_config['mongo_collection_obs']))
    try:
        _coll_usr = _db[_config['mongo_collection_pwd']]
        # cursor = coll.find()
        # for doc in cursor:
        #     print(doc)
        _logger.debug('Using collection {:s} with user access credentials in the database'.
                      format(_config['mongo_collection_pwd']))
    except Exception as _e:
        _coll_usr = None
        _logger.error(_e)
        _logger.error('Failed to use a collection {:s} with user access credentials in the database'.
                      format(_config['mongo_collection_pwd']))
    try:
        # build dictionary program num -> pi name
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
        _program_pi = None
        logger.error(_e)

    return _db, _coll, _program_pi


def get_fits_header(fits_file):
    """
        Get fits-file header
    :param fits_file:
    :return:
    """
    # read fits:
    with fits.open(os.path.join(fits_file)) as hdulist:
        # header:
        header = OrderedDict()
        for _entry in hdulist[0].header.cards:
            header[_entry[0]] = _entry[1:]

    return header


def check_pipe_automated(_config, _logger, _coll, _select, _date, _obs):
    """
        Check if observation has been automatically pipelined
    :param _config: config data
    :param _logger: logger instance
    :param _coll: collection in the database
    :param _select: database entry
    :param _date: date of obs
    :param _obs: obs name

    :return:
    """
    if not _select['pipelined']['automated']['status']['done']:
        # check if processed
        for tag in ('high_flux', 'faint', 'zero_flux', 'failed'):
            path_obs = os.path.join(_config['path_pipe'], _date, tag, _obs)
            if os.path.exists(path_obs):
                try:
                    # check folder modified date:
                    time_tag = datetime.datetime.utcfromtimestamp(
                        os.stat(path_obs).st_mtime)

                    fits100p = os.path.join(path_obs, '100p.fits')
                    header = get_fits_header(fits100p) if tag != 'failed' else {}

                    _coll.update_one(
                        {'_id': _obs},
                        {
                            '$set': {
                                'pipelined.automated.status.done': True,
                                'pipelined.automated.classified_as': tag,
                                'pipelined.automated.last_modified': time_tag,
                                'pipelined.automated.fits_header': header
                            },
                            '$push': {
                                'pipelined.automated.location': ['{:s}:{:s}'.format(
                                                    _config['analysis_machine_external_host'],
                                                    _config['analysis_machine_external_port']),
                                                    _config['path_pipe']],
                            }
                        }
                    )
                    _logger.debug('Updated automated pipeline entry for {:s}'.format(_obs))
                    # TODO: make preview images
                except Exception as _e:
                    print(_e)
                    _logger.error(_e)
                    return False
    # done? check modified time tag for updates:
    else:
        for tag in ('high_flux', 'faint', 'zero_flux', 'failed'):
            path_obs = os.path.join(_config['path_pipe'], _date, tag, _obs)
            if os.path.exists(path_obs):
                try:
                    # check folder modified date:
                    time_tag = datetime.datetime.utcfromtimestamp(
                        os.stat(path_obs).st_mtime)
                    # changed?
                    if _select['pipelined']['automated']['last_modified'] != time_tag:
                        fits100p = os.path.join(path_obs, '100p.fits')
                        header = get_fits_header(fits100p) if tag != 'failed' else {}

                        _coll.update_one(
                            {'_id': _obs},
                            {
                                '$set': {
                                    'pipelined.automated.classified_as': tag,
                                    'pipelined.automated.last_modified': time_tag,
                                    'pipelined.automated.fits_header': header
                                }
                            }
                        )
                        _logger.debug('Updated automated pipeline entry for {:s}'.format(_obs))
                        # TODO: remake preview images
                except Exception as _e:
                    print(_e)
                    _logger.error(_e)
                    return False

    return True


def check_pipe_faint(_config, _logger, _coll, _select, _date, _obs):
    """
        Check if observation has been processed
    :param _config: config data
    :param _logger: logger instance
    :param _coll: collection in the database
    :param _select: database entry
    :param _date: date of obs
    :param _obs: obs name

    :return:
    """
    if not _select['pipelined']['faint']['status']['done'] or \
                    _select['pipelined']['faint']['status']['retries'] < 3:
        # TODO: put job into the huey execution queue
        return False

    return True


def check_pipe_pca(_config, _logger, _coll, _select, _date, _obs):
    """
        Check if observation has been processed
        - when run for the first time, will put a task into the queue and retries++
        - when run for the second time, will generate preview images and mark as done
    :param _config: config data
    :param _logger: logger instance
    :param _coll: collection in the database
    :param _select: database entry
    :param _date: date of obs
    :param _obs: obs full name

    :return:
    """
    try:
        # not pipelined yet after <3 retries?
        if not _select['pipelined']['pca']['status']['done'] or \
                        _select['pipelined']['pca']['status']['retries'] < 3:
            # check if processed
            # name in accordance with the automated pipeline output,
            # but check the faint pipeline output folder too
            for tag in ('high_flux', 'faint'):
                path_obs = os.path.join(_config['path_pca'], _date, tag, _obs)
                # already processed?
                if os.path.exists(path_obs):
                    # check folder modified date:
                    time_tag = datetime.datetime.utcfromtimestamp(
                        os.stat(path_obs).st_mtime)
                    # load contrast curve
                    f_cc = [f for f in os.listdir(path_obs) if '_contrast_curve.txt' in f][0]
                    cc = np.loadtxt(f_cc)

                    # previews generated?
                    if not _select['pipelined']['pca']['status']['review']:
                        # load first image frame from a fits file
                        f_fits = [f for f in os.listdir(path_obs) if '.fits' in f][0]
                        _fits = load_fits(f_fits)
                        # scale with local contrast optimization for preview:
                        preview_img = scale_image(_fits, correction='local')

                        _status = generate_pca_images(_out_path=_config['path_pca'],
                                                      _sou_dir=_obs, _preview_img=preview_img,
                                                      _cc=cc, _fow_x=36, _pix_x=1024, _drizzle=True)

                        if _status:
                            _coll.update_one(
                                {'_id': _obs},
                                {
                                    '$set': {
                                        'pipelined.pca.status.preview': True
                                    }
                                }
                            )
                            _logger.debug('Updated pca pipeline entry [status.review] for {:s}'.format(_obs))
                        else:
                            _logger.debug('Failed to generate pca pipeline preview images for {:s}'.format(_obs))
                    # update database entry
                    _coll.update_one(
                        {'_id': _obs},
                        {
                            '$set': {
                                'pipelined.pca.status.done': True,
                                'pipelined.pca.contrast_curve': cc.tolist(),
                                'pipelined.pca.last_modified': time_tag
                            },
                            '$push': {
                                'pipelined.automated.location': ['{:s}:{:s}'.format(
                                    _config['analysis_machine_external_host'],
                                    _config['analysis_machine_external_port']),
                                    _config['path_pca']]
                            }
                        }
                    )
                    _logger.debug('Updated pca pipeline entry for {:s}'.format(_obs))
                # not processed yet? put a job into the queue then:
                else:
                    # this will produce a fits file with the psf-subtracted image
                    # and a text file with the contrast curve
                    # TODO:
                    job_pca(_obs, _logger, _coll)
                    _logger.debug('put a pca job into the queue for {:s}'.format(_obs))
                    # increment number of tries
                    _coll.update_one(
                        {'_id': _obs},
                        {
                            '$inc': {
                                'pipelined.pca.status.retries': 1
                            }
                        }
                    )
                    _logger.debug('Updated pca pipeline entry [status.retries] for {:s}'.format(_obs))

    except Exception as _e:
        print(_e)
        logger.error(_e)
        return False

    return True


def check_strehl(_config, _logger, _coll, _select, _date, _obs):
    """
        Check if observation has been processed
    :param _config: config data
    :param _logger: logger instance
    :param _coll: collection in the database
    :param _select: database entry
    :param _date: date of obs
    :param _obs: obs name

    :return:
    """
    if not _select['pipelined']['strehl']['status']['done'] or \
                    _select['pipelined']['strehl']['status']['retries'] < 3:
        job_strehl(_obs, _logger, _coll)
        print('put a Strehl job into the queue')
        return False

    return True


if __name__ == '__main__':
    """
        - create argument parser, parse command line arguments
        - set up logging
        - load config
    """

    ''' Create command line argument parser '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Data archive for Robo-AO')

    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)

    args = parser.parse_args()

    ''' set up logging '''
    logger = set_up_logging(_path='logs', _name='archive', _level=logging.DEBUG)

    # if you start me up... if you start me up I'll never stop (hopefully not)
    logger.info('Started daily archiving job.')

    try:
        ''' script absolute location '''
        abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))

        ''' load config data '''
        try:
            config = get_config(_config_file=args.config_file)
            logger.debug('Successfully read in the config file {:s}'.format(args.config_file))
        except Exception as e:
            logger.error(e)
            logger.error('Failed to read in the config file {:s}'.format(args.config_file))
            sys.exit()

        ''' Connect to the mongodb database '''
        try:
            db, coll, program_pi = connect_to_db(_config=config, _logger=logger)
            if None in (db, coll, program_pi):
                raise Exception('Failed to connect to the database')
        except Exception as e:
            logger.error(e)
            sys.exit()

        '''
         ###############################
         CHECK IF DATABASE IS UP TO DATE
         ###############################
        '''

        ''' check all raw data starting from config['archiving_start_date'] '''
        # get all dates with some raw data
        dates = [p for p in os.listdir(config['path_raw'])
                 if os.path.isdir(os.path.join(config['path_raw'], p))
                 and datetime.datetime.strptime(p, '%Y%m%d') >= config['archiving_start_date']]
        print(dates)
        # for each date get all unique obs names (used as _id 's in the db)
        for date in dates:
            date_files = os.listdir(os.path.join(config['path_raw'], date))
            # check the endings (\Z) and skip _N.fits.bz2:
            # must start with program number (e.g. 24_ or 24.1_)
            pattern_start = r'\d+.?\d??_'
            # must be a bzipped fits file
            pattern_end = r'.[0-9]{6}.fits.bz2\Z'
            pattern_fits = r'.fits.bz2\Z'
            # skip calibration files and pointings
            date_obs = [re.split(pattern_fits, s)[0] for s in date_files
                        if re.search(pattern_end, s) is not None and
                        re.match(pattern_start, s) is not None and
                        re.match('pointing_', s) is None and
                        re.match('bias_', s) is None and
                        re.match('dark_', s) is None and
                        re.match('flat_', s) is None and
                        re.match('seeing_', s) is None]
            print(date_obs)
            # TODO: handle seeing files separately [lower priority]
            date_seeing = [re.split(pattern_end, s)[0] for s in date_files
                           if re.search(pattern_end, s) is not None and
                           re.match('seeing_', s) is not None]
            print(date_seeing)
            # for each source name see if there's an entry in the database
            for obs in date_obs:
                print('processing {:s}'.format(obs))
                logger.debug('processing {:s}'.format(obs))
                # parse name:
                tmp = obs.split('_')
                # print(tmp)
                # program num
                prog_num = str(tmp[0])
                # who's pi?
                if prog_num in program_pi.keys():
                    prog_pi = program_pi[prog_num]
                else:
                    # play safe if pi's unknown:
                    prog_pi = 'admin'
                # stack name together if necessary (if contains underscores):
                sou_name = '_'.join(tmp[1:-5])
                # code of the filter used:
                filt = tmp[-4:-3][0]
                # date and time of obs:
                date_utc = datetime.datetime.strptime(tmp[-2] + tmp[-1], '%Y%m%d%H%M%S.%f')
                # camera:
                camera = tmp[-5:-4][0]
                # marker:
                marker = tmp[-3:-2][0]

                # look up entry in the database:
                select = coll.find_one({'_id': obs})
                # if entry not in database, create empty one and populate it
                if select is None:
                    print('{:s} not in database, adding...'.format(obs))
                    logger.info('{:s} not in database, adding'.format(obs))
                    entry = empty_db_record()
                    # populate:
                    entry['_id'] = obs
                    entry['name'] = sou_name
                    entry['science_program']['program_id'] = prog_num
                    entry['science_program']['program_PI'] = prog_pi
                    entry['date_utc'] = date_utc
                    if date_utc > datetime.datetime(2015, 10, 1):
                        entry['telescope'] = 'KPNO_2.1m'
                    else:
                        entry['telescope'] = 'Palomar_P60'
                    entry['camera'] = camera
                    entry['filter'] = filt  # also get this from FITS header

                    # find raw fits files:
                    raws = [s for s in date_files if re.match(obs, s) is not None]
                    entry['raw_data']['location'].append(['{:s}:{:s}'.format(
                                                    config['analysis_machine_external_host'],
                                                    config['analysis_machine_external_port']),
                                                    config['path_raw']])
                    entry['raw_data']['data'] = raws
                    entry['raw_data']['last_modified'] = datetime.datetime.now(pytz.utc)

                    # insert into database
                    result = coll.insert_one(entry)

                # entry found in database, check if pipelined, update entry if necessary
                else:
                    print('{:s} in database, checking...'.format(obs))
                    ''' check Nick-pipelined data '''
                    status_ok = check_pipe_automated(_config=config, _logger=logger, _coll=coll,
                                                     _select=select, _date=date, _obs=obs)
                    if not status_ok:
                        logger.error('Checking failed for automatic pipeline: {:s}'.format(obs))

                    ''' check Faint-pipelined data '''
                    status_ok = check_pipe_faint(_config=config, _logger=logger, _coll=coll,
                                                 _select=select, _date=date, _obs=obs)
                    if not status_ok:
                        logger.error('Checking failed for faint pipeline: {:s}'.format(obs))

                    ''' check (PCA-)pipelined data '''
                    status_ok = check_pipe_pca(_config=config, _logger=logger, _coll=coll,
                                               _select=select, _date=date, _obs=obs)
                    if not status_ok:
                        logger.error('Checking failed for PCA pipeline: {:s}'.format(obs))

                    ''' check Strehl data '''
                    status_ok = check_strehl(_config=config, _logger=logger, _coll=coll,
                                                     _select=select, _date=date, _obs=obs)
                    if not status_ok:
                        logger.error('Checking failed for automatic pipeline: {:s}'.format(obs))

                    ''' check seeing data '''
                    # TODO: [lower priority]
                    # for each date check if lists of processed and raw seeing files match
                    # rerun seeing.py for each date if necessary
                    # update last_modified if necessary

                    ''' check preview data '''
                    # for each date for each source check if processed
                    # collect jobs to execute on the way
                    # update last_modified if necessary

                # TODO: mark distributed when all pipelines done or n_retries>3,
                # TODO: compress everything with bzip2, store and transfer over to Caltech

            # TODO: query database for all contrast curves and Strehls [+seeing - lower priority]
            # TODO: make joint plots to display on the website

    except Exception as e:
        print(e)
        logger.error(e)
        logger.error('Unknown error.')
    finally:
        logger.info('Finished daily archiving job.')
