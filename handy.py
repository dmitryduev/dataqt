from __future__ import print_function
import os
from pymongo import MongoClient
import inspect
import ConfigParser
import datetime
import calendar
import argparse
import pyprind
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
# import seaborn as sns
#
# # set up plotting
# sns.set_style('whitegrid')
# plt.close('all')
# sns.set_context('talk')


def get_config(_config_file='config.ini'):
    """
        load config data
    """
    try:
        abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
        _config = ConfigParser.RawConfigParser()
        _config.read(os.path.join(abs_path, _config_file))

        ''' connect to mongodb database '''
        conf = dict()
        # paths:
        conf['path_archive'] = _config.get('Path', 'path_archive')
        # database access:
        conf['mongo_host'] = _config.get('Server', 'analysis_machine_external_host')
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


if __name__ == '__main__':
    ''' Create command line argument parser '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Manually talk to the Robo-AO database')

    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)

    args = parser.parse_args()
    config_file = args.config_file

    ''' get config data '''
    try:
        config = get_config(_config_file=config_file)
    except IOError:
        config = get_config(_config_file='config.ini')

    try:
        # connect to db, pull out collections:
        client, db, coll, coll_usr, coll_aux, coll_weather, program_pi = connect_to_db(config)

        # FIXME: YOUR CODE HERE!

        ''' filter wavelengths to scale seeing '''
        # seeing beta scales with lambda as follows: beta_2 = beta_1 * (lambda_1/lambda_2)**0.2
        # use central lambdas:
        filter_lambdas = {'Si': 763.0, 'Sz': 905.0, 'Sr': 622.0, 'Sg': 475.0, 'lp600': 770.0, 'c': 700.0}
        # scale factors to 500 nm:
        scale_factors = {f: (filter_lambdas[f]/500.0)**0.2 for f in filter_lambdas}

        ''' get aux data '''

        date_first_KP_light = datetime.datetime(2015, 12, 17)
        date_tcs_upgrade = datetime.datetime(2016, 10, 1)
        date_good_data = datetime.datetime(2017, 2, 21, 10, 0, 0)
        date_dec_amp_fixed = datetime.datetime(2017, 3, 1, 0, 0, 0)

        start = date_first_KP_light
        stop = datetime.datetime.utcnow()

        # get and plot only new (good-ish) stuff:
        # start = date_good_data
        # stop = datetime.datetime.utcnow()

        select_aux = coll_aux.find({'_id': {'$gte': start.strftime('%Y%m%d'), '$lt': stop.strftime('%Y%m%d')}})

        if select_aux.count() > 0:

            data_aux = []
            bar = pyprind.ProgBar(select_aux.count(), stream=1, title='Loading seeing data from aux collection',
                                  monitor=True)
            for ob in select_aux:
                # print('matching:', ob['_id'])
                bar.update(iterations=1)
                for frame in ob['seeing']['frames']:
                    if None not in frame:
                        # data_aux.append([frame[1]] + frame[3:])
                        scaled_seeing = [frame[3]*scale_factors[frame[2]], frame[4]*scale_factors[frame[2]],
                                         frame[5] * scale_factors[frame[2]]]
                        data_aux.append([frame[1]] + scaled_seeing + [frame[6]])

            data_aux = np.array(data_aux)
            # print(data_aux)

            # consider only measurements below 3 and above 0.5 arc seconds:
            mask_avg = np.all(np.vstack((data_aux[:, 1] > 0.5, data_aux[:, 1] < 3.0)), axis=0)
            mask_x = np.all(np.vstack((data_aux[:, 2] > 0.5, data_aux[:, 2] < 3.0)), axis=0)
            mask_y = np.all(np.vstack((data_aux[:, 3] > 0.5, data_aux[:, 3] < 3.0)), axis=0)

            print('median avg seeing: ', np.median(data_aux[mask_avg, 1]))
            print('median x seeing: ', np.median(data_aux[mask_x, 2]))
            print('median y seeing: ', np.median(data_aux[mask_y, 3]))

            fig = plt.figure('seeing hist')
            ax = fig.add_subplot(111)
            # the histogram of the seeing vs strehl data
            n, bins, patches = ax.hist(data_aux[mask_avg, 1], bins=50, normed=1, histtype='step',
                                       stacked=True, fill=False)
            # n, bins, patches = ax.hist(data_aux[mask_x, 2], bins=50, normed=1, histtype='step',
            #                            stacked=True, fill=False)
            # n, bins, patches = ax.hist(data_aux[mask_y, 3], bins=50, normed=1, histtype='step',
            #                            stacked=True, fill=False)

            # add percentiles:
            # median:
            median_seeing = np.median(data_aux[mask_avg, 1])
            q1_seeing = np.percentile(data_aux[mask_avg, 1], 25)
            q3_seeing = np.percentile(data_aux[mask_avg, 1], 75)
            plt.axvline(x=q1_seeing, label='25%: {:.2f}\"'.format(q1_seeing), color=plt.cm.Set1(0))
            plt.axvline(x=median_seeing, label='median: {:.2f}\"'.format(median_seeing), color=plt.cm.Set1(1))
            plt.axvline(x=q3_seeing, label='75%: {:.2f}\"'.format(q3_seeing), color=plt.cm.Set1(2))

            ax.set_xlabel('Seeing scaled to 500 nm [arc seconds]')
            ax.set_ylabel('Normalized counts')
            ax.grid(linewidth=0.5)
            ax.legend(loc='best', fancybox=True, prop={'size': 10})

        # dict to store query to be executed on the main collection (with obs data):
        query = dict()

        # query['filter'] = 'Si'

        query['seeing.median'] = {'$ne': None}

        query['date_utc'] = {'$gte': start, '$lt': stop}

        # exclude planetary data:
        query['science_program.program_id'] = {'$ne': '24'}

        # azimuth and elevation range:
        query['coordinates.azel.0'] = {'$gte': 0 * np.pi / 180.0, '$lte': 360 * np.pi / 180.0}
        query['coordinates.azel.1'] = {'$gte': 0 * np.pi / 180.0, '$lte': 90 * np.pi / 180.0}

        # consider reliable Strehls only:
        query['pipelined.automated.strehl.flag'] = {'$eq': 'OK'}

        # execute query:
        if len(query) > 0:
            # print('executing query:\n{:s}'.format(query))
            select = coll.find(query)

            print('total reliable Strehl measurements:', select.count())

            data = []

            # display a progress bar
            bar = pyprind.ProgBar(select.count(), stream=1, title='Loading query results', monitor=True)
            for ob in select:
                # print('matching:', ob['_id'])
                bar.update(iterations=1)
                data.append([ob['_id'], ob['date_utc'], ob['filter'],
                             ob['seeing']['nearest'][0], ob['seeing']['nearest'][1]*scale_factors[ob['filter']],
                             ob['pipelined']['automated']['strehl']['ratio_percent'],
                             ob['pipelined']['automated']['pca']['contrast_curve']])

            data = np.array(data)

            print('median Strehl: ', np.median(data[:, 5]))

            fig2 = plt.figure('seeing hist for science obsevations')
            ax2 = fig2.add_subplot(111)

            # the histogram of the seeing vs strehl data
            # exclude data taken more than 3 minutes before/after corresponding science observations:
            mask_real_close = np.abs(data[:, 3] - data[:, 1]) < datetime.timedelta(seconds=60*3)
            n, bins, patches = ax2.hist(data[mask_real_close, 4], bins=50, normed=1)

            # add percentiles:
            # median:
            median_seeing = np.median(data[mask_real_close, 4])
            q1_seeing = np.percentile(data[mask_real_close, 4], 25)
            q3_seeing = np.percentile(data[mask_real_close, 4], 75)
            print('For the selected data:')
            print('median: {:.2f}", 25%: {:.2f}", 75% {:.2f}"'.format(median_seeing, q1_seeing, q3_seeing))
            plt.axvline(x=q1_seeing, label='25%: {:.2f}\"'.format(q1_seeing), color=plt.cm.Pastel1(0))
            plt.axvline(x=median_seeing, label='median: {:.2f}\"'.format(median_seeing), color=plt.cm.Pastel1(1))
            plt.axvline(x=q3_seeing, label='75%: {:.2f}\"'.format(q3_seeing), color=plt.cm.Pastel1(2))

            ax2.set_xlabel('Seeing scaled to 500 nm [arc seconds]')
            ax2.set_ylabel('Normalized counts')
            # ax2.grid(linewidth=0.5)
            ax2.legend(loc='best', fancybox=True, prop={'size': 10})

            ''' more in-depth seeing '''
            total_nights = int((stop - start).days)
            print('total nights: {:d}'.format(total_nights))

            nights_with_seeing_data = sorted(list(set([d.date() for d in data[mask_real_close, 1]])))
            print('nights with seeing data: {:d}'.format(len(nights_with_seeing_data)))

            nightly_seeing = []
            for night in nights_with_seeing_data:
                data_night = [d.date() == night for d in data[:, 1]]
                mask_night_sci_seeing_meas = np.all(np.vstack((data_night, mask_real_close)), axis=0)

                seeing_night_median = np.median(data[mask_night_sci_seeing_meas, 4])
                nightly_seeing.append(seeing_night_median)
            nightly_seeing = np.array(nightly_seeing)

            # FIXME: switch
            plot_seeing_vs_time = False

            if plot_seeing_vs_time:
                fig21 = plt.figure('Seeing vs time')
                ax21 = fig21.add_subplot(111)
                ax21.plot(nights_with_seeing_data, nightly_seeing, '.', alpha=0.3, marker='o', markersize=4)
                ax21.grid(linewidth=0.5)

                # monthly violin plots:
                fig22 = plt.figure('Seeing vs month')
                ax22 = fig22.add_subplot(111)
                months = sorted(list(set([d.strftime('%Y%m') for d in nights_with_seeing_data])))
                months_ticks = sorted(list(set([d.strftime('%Y-%m') for d in nights_with_seeing_data])))

                violin = []
                print(months)
                for month in months:
                    mask_month = np.array([n.strftime('%Y%m') == month for n in nights_with_seeing_data])
                    violin.append(nightly_seeing[mask_month])

                # add number of nights with data to ticks:
                for ii, tick in enumerate(months_ticks):
                    months_ticks[ii] += '\n{:d}/{:d}'.format(len(violin[ii]),
                                                             calendar.monthrange(int(tick[:4]), int(tick[5:]))[1])

                ax22.violinplot(violin, showmeans=False, showmedians=True, showextrema=True, points=100)
                ax22.set_xticks([y+1 for y in range(len(violin))])
                ax22.set_xticklabels(months_ticks)
                ax22.grid(linewidth=0.5)

                import seaborn as sns
                sns.set(style='whitegrid')
                fig23 = plt.figure('Seeing vs month violin')
                ax23 = fig23.add_subplot(211)
                sns.violinplot(data=violin, inner='quartile', cut=0, scale_hue=False,
                               color='steelblue', scale='count', linewidth=1, bw=.2)
                # sns.violinplot(data=violin, inner='quartile', cut=2,
                #                color=plt.cm.Blues(0.5), scale='count', linewidth=1)
                # sns.despine(left=True)
                ax23.set_xticks([y for y in range(len(violin))])
                ax23.set_xticklabels(months_ticks)
                ax23.set_ylabel('Seeing, arc seconds')

                # fig24 = plt.figure('Seeing vs month box')
                ax24 = fig23.add_subplot(212)
                sns.boxplot(data=violin, color=plt.cm.Blues(0.5), linewidth=1)
                ax24.set_xticks([y for y in range(len(violin))])
                ax24.set_xticklabels(months_ticks)
                ax24.set_ylabel('Seeing, arc seconds')

            ''' Strehl vs seeing '''
            fig3 = plt.figure('Strehl vs seeing')

            # FIXME: switch
            plot_new_good_data = True

            if plot_new_good_data:
                ax3 = fig3.add_subplot(121)
                ax3_new = fig3.add_subplot(122)
            else:
                ax3 = fig3.add_subplot(111)

            filters = set(data[:, 2])
            for filt in filters:
                mask_filter = np.array(data[:, 2] == filt)

                print('median Strehl in {:s}: {:.1f}'.format(filt, np.median(data[mask_filter, 5])))

                ax3.plot(data[mask_filter, 4], data[mask_filter, 5], '.', alpha=0.3,
                         marker='o', markersize=4, label=filt)

                if plot_new_good_data:
                    # mask_new_data = np.array(data[:, 1] > date_tcs_upgrade)
                    mask_new_data = np.array(data[:, 1] > date_good_data)
                    # print(np.count_nonzero(mask_new_data))
                    mask_filter_new_data = np.all(np.vstack((mask_filter, mask_new_data)), axis=0)
                    # print(np.count_nonzero(mask_filter_new_data), '\n')
                    ax3_new.plot(data[mask_filter_new_data, 4], data[mask_filter_new_data, 5], '.', alpha=0.3,
                                 marker='o', markersize=4, label=filt)

            ax3.set_xlabel('Seeing [arc seconds]')
            ax3.set_ylabel('Strehl ratio, %')
            ax3.legend(loc='best', numpoints=1, fancybox=True, prop={'size': 10})

            if plot_new_good_data:
                ax3_new.set_xlabel('Seeing [arc seconds]')
                ax3_new.set_ylabel('Strehl ratio, %')
                ax3_new.legend(loc='best', numpoints=1, fancybox=True, prop={'size': 10})

            fig4 = plt.figure('contrast curves')
            ax4 = fig4.add_subplot(111)

            sep_mean = np.linspace(0.2, 1.45, num=100)

            for filt in filters:
                mask_filter = np.array(data[:, 2] == filt)

                cc_mean = []

                for entry in data[mask_filter, :]:
                    if entry[6] is not None and len(entry[6]) > 0:
                        cc = np.array(entry[6])
                    else:
                        continue

                    cc_tmp = np.interp(sep_mean, cc[:, 0], cc[:, 1])
                    if not np.isnan(cc_tmp).any():
                        cc_mean.append(cc_tmp)

                cc_mean = np.array(cc_mean)
                # print(cc_mean.shape)

                # best 10%:
                cc_mean_val = np.mean(cc_mean, axis=1)
                best_10p = np.percentile(cc_mean_val, 90)

                mask_best_10p = cc_mean_val > best_10p

                baseline, = ax4.plot(sep_mean, np.median(cc_mean.T, axis=1), '-', linewidth=1.9, label=filt)
                ax4.plot(sep_mean, np.median(cc_mean.T[:, mask_best_10p], axis=1), '--', linewidth=1.2,
                         label='{:s}, best 10%'.format(filt), c=baseline.get_color())

            ax4.set_xlim([0.2, 1.45])
            ax4.set_xlabel('Separation [arc seconds]')  # , fontsize=18)
            ax4.set_ylabel('5-sigma contrast [$\Delta$mag]')  # , fontsize=18)
            ax4.set_ylim([0, 8])
            ax4.set_ylim(ax4.get_ylim()[::-1])
            ax4.grid(linewidth=0.5)
            ax4.legend(loc='best', numpoints=1, fancybox=True, prop={'size': 6})

    except Exception as e:
        print(e)

    finally:
        plt.show()

        client.close()
