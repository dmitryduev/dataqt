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
import traceback
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from astropy.io import fits

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
        conf['mongo_replicaset'] = _config.get('Database', 'replicaset')

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
        _client = MongoClient(host=_config['mongo_host'], port=_config['mongo_port'],
                              replicaset=_config['mongo_replicaset'])
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

        date_first_KP_light = datetime.datetime(2015, 12, 17).replace(tzinfo=pytz.utc)
        date_tcs_upgrade = datetime.datetime(2016, 10, 1).replace(tzinfo=pytz.utc)
        date_good_data = datetime.datetime(2017, 2, 21, 10, 0, 0).replace(tzinfo=pytz.utc)
        date_dec_amp_fixed = datetime.datetime(2017, 3, 1, 0, 0, 0).replace(tzinfo=pytz.utc)

        # start = date_first_KP_light
        # stop = (datetime.datetime.utcnow() + datetime.timedelta(days=1)).replace(tzinfo=pytz.utc)

        # get and plot only new (good-ish) stuff:
        start = date_good_data
        stop = (datetime.datetime.utcnow() + datetime.timedelta(days=1)).replace(tzinfo=pytz.utc)

        select_aux = coll_aux.find({'_id': {'$gte': start.strftime('%Y%m%d'), '$lt': stop.strftime('%Y%m%d')}})

        if select_aux.count() > 0:

            data_aux = []
            psflib_ids = []
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
                # get obs ids that are in the PSF library:
                for _id in ob['psf_lib']:
                    if ob['psf_lib'][_id]['in_lib']:
                        psflib_ids.append(_id)

            data_aux = np.array(data_aux)
            psflib_ids = np.array(psflib_ids)
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

        # # load PSF library:
        # from astropy.io import fits
        # with fits.open('/Users/dmitryduev/_caltech/roboao/_archive/psf_library.fits') as _lib:
        #     _library = _lib[0].data
        #     psflib_ids = _lib[-1].data['obs_names']

        # dict to store query to be executed on the main collection (with obs data):
        query = dict()

        # query['filter'] = 'Si'

        query['seeing.median'] = {'$ne': None}

        query['date_utc'] = {'$gte': start, '$lt': stop}

        # only grab observations with updated contrast curves:
        # query['pipelined.automated.pca.last_modified'] = {'$gte':
        #                                                   datetime.datetime(2017, 3, 16).replace(tzinfo=pytz.utc)}

        # get stuff

        # exclude planetary data:
        query['science_program.program_id'] = {'$ne': '24'}

        # azimuth and elevation range:
        query['coordinates.azel.0'] = {'$gte': 0 * np.pi / 180.0, '$lte': 360 * np.pi / 180.0}
        query['coordinates.azel.1'] = {'$gte': 0 * np.pi / 180.0, '$lte': 90 * np.pi / 180.0}

        # consider reliable Strehls only:
        # query['pipelined.automated.strehl.flag'] = {'$eq': 'OK'}

        # discard observations marked as "zero_flux" by the automated pipeline
        # query['pipelined.automated.classified_as'] = {'$ne': 'zero_flux'}
        # query['pipelined.automated.classified_as'] = {'$nin': ['zero_flux', 'failed', 'faint']}
        # query['pipelined.automated.classified_as'] = {'$nin': ['zero_flux', 'failed']}
        query['pipelined.automated.classified_as'] = {'$nin': ['failed']}

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
                # FIXME: only get stuff that is in the PSF library:
                if ob['_id'].replace('.', '_') not in psflib_ids:
                    # continue
                    pass

                bar.update(iterations=1)
                # correct seeing for Zenith distance and reference to 500 nm
                data.append([ob['_id'], ob['date_utc'], ob['filter'],
                             ob['seeing']['nearest'][0],
                             ob['seeing']['nearest'][1]*scale_factors[ob['filter']] *
                                (np.cos(np.pi/2 - ob['coordinates']['azel'][1])**0.6),
                             ob['pipelined']['automated']['strehl']['ratio_percent'],
                             ob['pipelined']['automated']['pca']['contrast_curve'],
                             ob['coordinates']['azel'][1]*180/np.pi,
                             ob['magnitude'],
                             ob['pipelined']['automated']['strehl']['flag']
                             ])

            data = np.array(data)

            print('\nloaded {:d} observations'.format(data.shape[0]))

            print('median elevation: ', np.median(data[:, 7]))

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

            median_seeing = np.median(nightly_seeing)
            q1_seeing = np.percentile(nightly_seeing, 25)
            q3_seeing = np.percentile(nightly_seeing, 75)
            print('For the nightly median values:')
            print('median: {:.2f}", 25%: {:.2f}", 75% {:.2f}"'.format(median_seeing, q1_seeing, q3_seeing))

            # the histogram of the seeing vs strehl data
            fig20 = plt.figure('seeing hist for nightly median values')
            ax20 = fig20.add_subplot(111)
            n, bins, patches = ax20.hist(nightly_seeing, bins=20, normed=1)

            # add percentiles:
            plt.axvline(x=q1_seeing, label='25%: {:.2f}\"'.format(q1_seeing), color=plt.cm.Pastel1(0))
            plt.axvline(x=median_seeing, label='median: {:.2f}\"'.format(median_seeing), color=plt.cm.Pastel1(1))
            plt.axvline(x=q3_seeing, label='75%: {:.2f}\"'.format(q3_seeing), color=plt.cm.Pastel1(2))

            ax20.set_xlabel('Seeing scaled to 500 nm [arc seconds]')
            ax20.set_ylabel('Normalized counts')
            # ax2.grid(linewidth=0.5)
            ax20.legend(loc='best', fancybox=True, prop={'size': 10})

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
                # sns.set(style='whitegrid')
                with sns.axes_style("whitegrid"):
                    fig23 = plt.figure('Seeing vs month violin')
                    ax23 = fig23.add_subplot(211)
                    sns.violinplot(data=violin, inner='quartile', cut=0, scale_hue=False,
                                   color=plt.cm.Blues(0.5), scale='count', linewidth=1.7, bw=.2)
                    # color='steelblue'
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
            plot_new_good_data = False

            if plot_new_good_data:
                ax3 = fig3.add_subplot(121)
                ax3_new = fig3.add_subplot(122)

                fig31 = plt.figure('Strehl vs seeing: old vs new')
                ax31 = fig31.add_subplot(111)
            else:
                ax3 = fig3.add_subplot(111)

            filters = set(data[:, 2])
            for filt in filters:
                mask_filter = np.array(data[:, 2] == filt)

                # make sure to exclude NaNs:
                seeing_filt = np.array(data[:, 4], dtype=np.float)
                strehl_filt = np.array(data[:, 5], dtype=np.float)
                # print(np.min(seeing_filt), np.max(seeing_filt), np.nan in seeing_filt, np.inf in seeing_filt)
                # print(np.min(strehl_filt), np.max(strehl_filt), np.nan in strehl_filt, np.inf in strehl_filt)
                mask_not_a_nan = np.logical_not(np.isnan(strehl_filt))

                mask_filter = np.all(np.vstack((mask_filter, mask_not_a_nan)), axis=0)

                print('median Strehl in {:s}: {:.1f}'.format(filt, np.median(data[mask_filter, 5])))

                # ax3.plot(data[mask_filter, 4], data[mask_filter, 5], '.', alpha=0.3,
                #          marker='o', markersize=4, label=filt)
                # when there's not much data:
                ax3.plot(data[mask_filter, 4], data[mask_filter, 5], '.', alpha=0.5,
                         marker='o', markersize=4, label=filt)
                # ax3.plot(data[mask_filter, 4], data[mask_filter, 5], '.', alpha=0.6,
                #          marker='o', markersize=5, label=filt)

                if plot_new_good_data:
                    # mask_new_data = np.array(data[:, 1] > date_tcs_upgrade)
                    mask_new_data = np.array(data[:, 1] > date_good_data)
                    # print(np.count_nonzero(mask_new_data))
                    mask_filter_new_data = np.all(np.vstack((mask_filter, mask_new_data)), axis=0)
                    # print(np.count_nonzero(mask_filter_new_data), '\n')
                    ax3_new.plot(data[mask_filter_new_data, 4], data[mask_filter_new_data, 5], '.', alpha=0.3,
                                 marker='o', markersize=4, label=filt)

                    # use same colors when plotted on the same plot:
                    # baseline, = ax31.plot(data[mask_filter, 4], data[mask_filter, 5], '.', alpha=0.2,
                    #                       marker='o', markersize=4, label=filt)
                    # ax31.plot(data[mask_filter_new_data, 4], data[mask_filter_new_data, 5], '.', alpha=0.7,
                    #           marker='o', markersize=5, label=filt, c=baseline.get_color())
                    # use different colors when plotted on the same plot:
                    baseline_old, = ax31.plot(data[mask_filter, 4], data[mask_filter, 5], '.', alpha=0.2,
                                              marker='o', markersize=4, label=filt)
                    baseline_new, = ax31.plot(data[mask_filter_new_data, 4], data[mask_filter_new_data, 5],
                                              '.', alpha=0.5, marker='o', markersize=5, label=filt)

                    if True:
                        # for better visual apprehension, make robust RANSAC fits to old/new data
                        seeing_for_ransac_old = np.expand_dims(seeing_filt[mask_filter], axis=1)
                        seeing_for_ransac_new = np.expand_dims(seeing_filt[mask_filter_new_data], axis=1)
                        # seeing_for_ransac_old = data[mask_filter, 4]
                        # seeing_for_ransac_new = data[mask_filter_new_data, 4]
                        estimators = [('RANSAC', linear_model.RANSACRegressor()), ]
                        for name, estimator in estimators:
                            model = make_pipeline(PolynomialFeatures(degree=5), estimator)
                            model.fit(seeing_for_ransac_old, np.expand_dims(strehl_filt[mask_filter], axis=0))
                            strehl_ransac_old = model.predict(seeing_for_ransac_old)
                            model = make_pipeline(PolynomialFeatures(degree=5), estimator)
                            model.fit(seeing_for_ransac_new, np.expand_dims(strehl_filt[mask_filter_new_data], axis=0))
                            strehl_ransac_new = model.predict(seeing_for_ransac_new)

                            ax31.plot(data[mask_filter, 4], strehl_ransac_old, '--', c=baseline_old.get_color(),
                                      linewidth=1, label='Robust {:s} fit'.format(name), clip_on=True)
                            ax31.plot(data[mask_filter_new_data, 4], strehl_ransac_new, '--', c=baseline_new.get_color(),
                                      linewidth=1, label='Robust {:s} fit'.format(name), clip_on=True)

            ax3.set_xlabel('Seeing [arc seconds]')
            ax3.set_ylabel('Strehl ratio, %')
            ax3.legend(loc='best', numpoints=1, fancybox=True, prop={'size': 10})

            if plot_new_good_data:
                ax3_new.set_xlabel('Seeing [arc seconds]')
                ax3_new.set_ylabel('Strehl ratio, %')
                ax3_new.legend(loc='best', numpoints=1, fancybox=True, prop={'size': 10})

                ax31.set_xlabel('Seeing [arc seconds]')
                ax31.set_ylabel('Strehl ratio, %')
                ax31.legend(loc='best', numpoints=1, fancybox=True, prop={'size': 10})

            fig4 = plt.figure('contrast curves', figsize=(10, 7))
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
                # make sure to use the same color for the best 10% cases
                ax4.plot(sep_mean, np.median(cc_mean.T[:, mask_best_10p], axis=1), '--', linewidth=1.2,
                         label='{:s}, best 10%'.format(filt), c=baseline.get_color())

            ax4.set_xlim([0.2, 1.45])
            ax4.set_xlabel('Separation [arc seconds]', fontsize=18)
            ax4.set_ylabel('5-sigma contrast [$\Delta$mag]', fontsize=18)
            ax4.set_ylim([0, 8])
            ax4.set_ylim(ax4.get_ylim()[::-1])
            ax4.grid(linewidth=0.5)
            # ax4.legend(loc='best', numpoints=1, fancybox=True, prop={'size': 16})
            ax4.legend(loc='best', numpoints=1, fancybox=True, prop={'size': 10})

            ''' Strehl vs guide star mag '''
            if False:
                fig5 = plt.figure('Strehl vs guide star mag')
                ax5 = fig5.add_subplot(111)

                # exclude data taken more than 3 minutes before/after corresponding science observations:
                mask_real_close = np.abs(data[:, 3] - data[:, 1]) < datetime.timedelta(seconds=60 * 3)
                mask_i = np.array(data[:, 2] == u'Si')
                mask_lp600 = np.array(data[:, 2] == u'lp600')
                mask_ok = np.array(data[:, 9] == u'OK')
                mask_bad = np.array(data[:, 9] == u'BAD?')
                mask_no_bright = data[:, 8] > 10

                # print(np.max(mask_i), np.max(mask_ok), )
                # print(data[:, 8])

                for seeing in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8):

                    mask_median_seeing = np.abs(data[:, 4] - seeing) < 0.1

                    mask_master_ok = np.all(np.vstack((mask_real_close, mask_no_bright, mask_i,
                                                       mask_ok, mask_median_seeing)), axis=0)
                    mask_master_bad = np.all(np.vstack((mask_real_close, mask_no_bright, mask_i,
                                                        mask_bad, mask_median_seeing)), axis=0)

                    baseline, = ax5.plot(data[mask_master_ok, 8], data[mask_master_ok, 5], '.', alpha=0.2,
                             marker='o', markersize=4, label='Si at {:.1f}\'\''.format(seeing))
                    ax5.plot(data[mask_master_bad, 8], data[mask_master_bad, 5], '.', alpha=0.6,
                             marker='o', markersize=5, label='Si_bad at {:.1f}\'\''.format(seeing),
                             c=baseline.get_color())

                ax5.set_xlabel('Magnitude')
                ax5.set_ylabel('Strehl ratio, %')
                ax5.grid(linewidth=0.5)
                ax5.legend(loc='best', numpoints=1, fancybox=True, prop={'size': 10})

    except Exception as e:
        traceback.print_exc()
        print(e)

    finally:
        plt.show()

        client.close()
