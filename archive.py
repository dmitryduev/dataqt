"""
    Data Archiving for Robo-AO

    Generates stuff to be displayed on the archive
    Updates the database

    DAD (Caltech) 2016
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

        '''
         ###############################
         CHECK IF DATABASE IS UP TO DATE
         ###############################
        '''

        ''' check all raw data from, say, January 2016 '''
        # get all dates
        # for each date get all source names
        # skip calibration files and pointings
        # TODO: handle seeing files separately [lower priority]
        # for each source name see if there's an entry in the database
        # if not, create an empty one
        # if yes, check further

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


    finally:
        logger.info('Finished daily archiving job.')
