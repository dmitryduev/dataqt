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
from astropy.io import fits
from pymongo import MongoClient
import sys
import re
from collections import OrderedDict


def empty_db_record():
    return {
            '_id': None,
            'date_added': datetime.datetime.now(),
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
                        'done': False
                    },
                    'location': [],
                    'classified_as': None,
                    'fits_header': {},
                    'last_modified': datetime.datetime.now()
                },
                'faint': {
                    'status': {
                        'done': False,
                        'retries': 0
                    },
                    'location': [],
                    'last_modified': datetime.datetime.now()
                },
                'pca': {
                    'status': {
                        'done': False,
                        'retries': 0
                    },
                    'location': [],
                    'contrast_curve': {},
                    'last_modified': datetime.datetime.now()
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
                    'last_modified': datetime.datetime.now()
                }
            },

            'seeing': {
                'median': None,
                'mean': None,
                'last_modified': datetime.datetime.now()
            },
            'bzip2': {
                'location': [],
                'last_modified': datetime.datetime.now()
            },
            'raw_data': {
                'location': [],
                'data': [],
                'last_modified': datetime.datetime.now()
            },
            'comment': None
        }


def set_up_logging(_path='logs', _name='archive', _level=logging.DEBUG):
    """ Set up logging

    :param _path:
    :return: logger instance
    """

    if not os.path.exists(_path):
        os.makedirs(_path)
    utc_now = datetime.datetime.utcnow()

    # http://www.blog.pythonlibrary.org/2012/08/02/python-101-an-intro-to-logging/
    _logger = logging.getLogger(_name)
    # logger.setLevel(logging.INFO)
    _logger.setLevel(_level)
    # create the logging file handler
    fh = logging.FileHandler(os.path.join(_path,
                                          '{:s}.{:s}.log'.format(_name, utc_now.strftime('%Y%m%d'))),
                             mode='w')

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


if __name__ == '__main__':

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
                    print('{:s} not in database, adding'.format(obs))
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

                    # insert into database
                    result = coll.insert_one(entry)

                # entry in database, check if pipelined
                else:
                    print('{:s} in database, checking'.format(obs))
                    ''' check Nick-pipelined data '''
                    if not select['pipelined']['automated']['status']['done']:
                        # for each date for each source check if processed
                        for tag in ('high_flux', 'faint', 'zero_flux', 'failed'):
                            path_obs = os.path.join(config['path_pipe'], date, tag, obs)
                            if os.path.exists(path_obs):
                                try:
                                    # check folder modified date:
                                    time_tag = datetime.datetime.utcfromtimestamp(
                                                    os.stat(path_obs).st_mtime)

                                    fits100p = os.path.join(path_obs, '100p.fits')
                                    header = get_fits_header(fits100p) if tag != 'failed' else {}

                                    result = coll.update_one(
                                        {'_id': obs},
                                        {
                                            '$set': {
                                                'pipelined.automated.status.done': True,
                                                'pipelined.automated.classified_as': tag,
                                                'pipelined.automated.location':
                                                    select['pipelined']['automated']['location'].append(
                                                        ['{:s}:{:s}'.format(
                                                            config['analysis_machine_external_host'],
                                                            config['analysis_machine_external_port']),
                                                            config['path_pipe']]
                                                    ),
                                                'pipelined.automated.last_modified': time_tag,
                                                'pipelined.automated.fits_header': header
                                            }
                                        }
                                    )
                                    logger.debug('Updated automated pipeline entry for {:s}'.format(obs))
                                except Exception as e:
                                    print(e)
                                    logger.error(e)
                    # done? check modified time tag for updates:
                    else:
                        for tag in ('high_flux', 'faint', 'zero_flux', 'failed'):
                            path_obs = os.path.join(config['path_pipe'], date, tag, obs)
                            if os.path.exists(path_obs):
                                try:
                                    # check folder modified date:
                                    time_tag = datetime.datetime.utcfromtimestamp(
                                                    os.stat(path_obs).st_mtime)
                                    # changed?
                                    if select['pipelined']['automated']['last_modified'] != time_tag:
                                        fits100p = os.path.join(path_obs, '100p.fits')
                                        header = get_fits_header(fits100p) if tag != 'failed' else {}

                                        result = coll.update_one(
                                            {'_id': obs},
                                            {
                                                '$set': {
                                                    'pipelined.automated.classified_as': tag,
                                                    'pipelined.automated.last_modified': time_tag,
                                                    'pipelined.automated.fits_header': header
                                                }
                                            }
                                        )
                                        logger.debug('Updated automated pipeline entry for {:s}'.format(obs))
                                except Exception as e:
                                    print(e)
                                    logger.error(e)

                    ''' check Faint-pipelined data '''
                    # for each date for each source check if processed
                    # update last_modified if necessary

                    ''' check (PCA-)pipelined data '''
                    # for each date for each source check if processed
                    # collect jobs to execute on the way
                    # update last_modified if necessary

                    ''' check Strehl data '''
                    # for each date for each source check if processed
                    # collect jobs to execute on the way
                    # update last_modified if necessary

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

    except Exception as e:
        print(e)
        logger.error(e)
        logger.error('Unknown error.')
    finally:
        logger.info('Finished daily archiving job.')
