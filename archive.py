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
from pymongo import MongoClient
import sys
import re


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


if __name__ == '__main__':

    ''' Create command line argument parser '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Data archive for Robo-AO')

    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)

    args = parser.parse_args()

    try:
        ''' Set up logging '''
        if not os.path.exists('logs'):
            os.makedirs('logs')
        utc_now = datetime.datetime.utcnow()

        # http://www.blog.pythonlibrary.org/2012/08/02/python-101-an-intro-to-logging/
        logger = logging.getLogger('archive')
        # logger.setLevel(logging.INFO)
        logger.setLevel(logging.DEBUG)
        # create the logging file handler
        fh = logging.FileHandler(os.path.join('logs',
                                              'archive.{:s}.log'.format(utc_now.strftime('%Y%m%d'))),
                                 mode='w')

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # formatter = logging.Formatter('%(asctime)s %(message)s')
        fh.setFormatter(formatter)

        # add handler to logger object
        logger.addHandler(fh)

        # if you start me up... if you start me up I'll never stop (hopefully not)
        logger.info('Started daily archiving job.')

        ''' script absolute location '''
        abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))

        ''' Get config data '''
        # load config data
        config = ConfigParser.RawConfigParser()

        if args.config_file[0] not in ('/', '~'):
            if os.path.isfile(os.path.join(abs_path, args.config_file)):
                config.read(os.path.join(abs_path, args.config_file))
                if len(config.read(os.path.join(abs_path, args.config_file))) == 0:
                    raise Exception('Failed to load config file')
            else:
                raise IOError('Failed to find config file')
        else:
            if os.path.isfile(args.config_file):
                config.read(args.config_file)
                if len(config.read(args.config_file)) == 0:
                    raise Exception('Failed to load config file')
            else:
                raise IOError('Failed to find config file')

        # planetary program number (do no crop planetary images!)
        program_num_planets = int(config.get('Programs', 'planets'))
        # path to raw data:
        path_raw = config.get('Path', 'path_raw')
        # path to (standard) pipeline data:
        path_pipe = config.get('Path', 'path_pipe')
        # path to Becky-pipeline data:
        path_pca = config.get('Path', 'path_pca')
        # path to seeing plots:
        path_seeing = config.get('Path', 'path_seeing')
        # website data dwelling place:
        path_to_website_data = config.get('Path', 'path_to_website_data')

        # database access:
        mongo_host = config.get('Database', 'host')
        mongo_port = int(config.get('Database', 'port'))
        mongo_db = config.get('Database', 'db')
        mongo_collection_obs = config.get('Database', 'collection_obs')
        mongo_user = config.get('Database', 'user')
        mongo_pwd = config.get('Database', 'pwd')

        logger.debug('Successfully read in the config file {:s}'.format(args.config_file))

        ''' Connect to the mongodb database '''
        try:
            client = MongoClient(host=mongo_host, port=mongo_port)
            db = client[mongo_db]
            logger.debug('Successfully connected to the Robo-AO database at {:s}:{:d}'.
                         format(mongo_host, mongo_port))
        except Exception as e:
            logger.error(e)
            logger.error('Failed to connect to the Robo-AO database at {:s}:{:d}'.
                         format(mongo_host, mongo_port))
            sys.exit()
        try:
            db.authenticate(mongo_user, mongo_pwd)
            logger.debug('Successfully authenticated with the Robo-AO database at {:s}:{:d}'.
                         format(mongo_host, mongo_port))
        except Exception as e:
            logger.error(e)
            logger.error('Authentication failed for the Robo-AO database at {:s}:{:d}'.
                         format(mongo_host, mongo_port))
            sys.exit()
        try:
            coll = db[mongo_collection_obs]
            # cursor = coll.find()
            # for doc in cursor:
            #     print(doc)
            logger.debug('Using collection {:s} with obs data in the database'.
                         format(mongo_collection_obs))
        except Exception as e:
            logger.error(e)
            logger.error('Failed to use a collection {:s} with obs data in the database'.
                         format(mongo_collection_obs))
            sys.exit()

        '''
         ###############################
         CHECK IF DATABASE IS UP TO DATE
         ###############################
        '''

        ''' check all raw data from, say, January 2016 '''
        # get all dates with some raw data
        dates = [p for p in os.listdir(path_raw)
                 if os.path.isdir(os.path.join(path_raw, p))]
        print(dates)
        # for each date get all unique obs names (used as _id 's in the db)
        for date in dates:
            date_files = os.listdir(os.path.join(path_raw, date))
            # check the endings (\Z) and skip _N.fits.bz2:
            # must start with program number (e.g. 24_ or 24.1_)
            pattern_start = r'\d+.?\d??_'
            # must be a bzipped fits file
            pattern_end = r'.[0-9]{6}.fits.bz2\Z'
            # skip calibration files and pointings
            date_obs = [re.split(pattern_end, s)[0] for s in date_files
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
            # TODO: for each source name see if there's an entry in the database
            for obs in date_obs:
                print('processing {:s}'.format(obs))
                logger.debug('processing {:s}'.format(obs))
                # parse name:
                tmp = obs.split('_')
                # program num
                prog_num = int(tmp[0])
                # stack name together if necessary (if contains underscores):
                sou_name = '_'.join(tmp[1:-5])
                # code of the filter used:
                filt = tmp[-4:-3][0]
                # date and time of obs:
                time = datetime.datetime.strptime(tmp[-2] + tmp[-1], '%Y%m%d%H%M%S.%f')
                # camera:
                camera = tmp[-5:-4][0]
                # marker:
                marker = tmp[-3:-2][0]
                # look up entry in the database:
                select = coll.find_one({'_id': obs})
                # if entry not in database, create an empty one and populate it
                if select is None:
                    print('{:s} not in database, adding'.format(obs))
                    logger.info('{:s} not in database, adding'.format(obs))
                    entry = empty_db_record()
                    # populate:
                    entry['_id'] = obs
                    entry['name'] = sou_name


            # TODO: if yes, check further

        ''' check Nick-pipelined data '''
        # for each date for each source check if processed
        # update last_modified if necessary

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
    except:
        logger.error('Unknown error.')
    finally:
        logger.info('Finished daily archiving job.')
