from __future__ import print_function
import inspect
from astropy.io import fits
import json
import argparse
import os
import shutil
import datetime
import numpy as np
# from bokeh.io import gridplot, output_file, show
# from bokeh.plotting import figure
import sewpy
from skimage import exposure, img_as_float
from skimage.transform import rescale
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker, TextArea
import seaborn as sns
sns.set_style('whitegrid')
# sns.set_palette(sns.diverging_palette(10, 220, sep=80, n=7))
plt.close('all')
sns.set_context('talk')


# python -W ignore dataqt.py /Users/dmitryduev/_caltech/roboao/_auto_reductions/
#                            /Users/dmitryduev/_caltech/roboao/seeing/
#                            --date 20160314

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
        self.size_bar.add_artist(Rectangle((0, 0), size, 0, fc='none', color='white'))

        self.txt_label = TextArea(label, dict(color='white'), minimumdescent=False)

        self._box = VPacker(children=[self.size_bar, self.txt_label],
                            align="center",
                            pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=self._box,
                                   prop=prop,
                                   frameon=frameon)


def make_img(_sou_name, _time, _filter, _prog_num, _path_sou, _path_data, pipe_out_type):
    """

    :param _sou_name:
    :param _time:
    :param _filter:
    :param _prog_num:
    :param _path_sou:
    :param _path_data:
    :param pipe_out_type: 'high_flux' or 'faint'
    :return:
    """
    # read, process and display fits:
    hdulist = fits.open(os.path.join(_path_sou, '100p.fits'))
    scidata = hdulist[0].data

    # extract sources:
    sew = sewpy.SEW(
            # params=["X_IMAGE", "Y_IMAGE", "X2_IMAGE", "Y2_IMAGE",
            #         "A_IMAGE", "B_IMAGE", "FLUX_APER(3)", "FLAGS",
            #         "FWHM_IMAGE", "VIGNET"],
            params=["X_IMAGE", "Y_IMAGE", "XPEAK_IMAGE", "YPEAK_IMAGE",
                    "A_IMAGE", "B_IMAGE",
                    "FWHM_IMAGE", "FLAGS"],
            config={"DETECT_MINAREA": 10, "PHOT_APERTURES": "10", 'DETECT_THRESH': '5.0'},
            sexpath="sex")
    # By default, this assumes that SExtractor can be called as "sex"
    # If this is not the case, or if the executable is not in your path,
    # specify the path by adding the argument sexpath="/path/to/sextractor"
    # to the above instantiation.

    out = sew(os.path.join(_path_sou, '100p.fits'))
    # sort according to FWHM
    out['table'].sort('FWHM_IMAGE')
    # descending order
    out['table'].reverse()

    # print(out['table'])  # This is an astropy table.

    # create a plot

    # bokeh
    # s = figure(x_range=(0, 10), y_range=(0, 10), width=700, plot_height=700, title=None)
    # s.image(image=[scidata], x=0, y=0, dw=10, dh=10, palette="Spectral11")
    # show(s)
    # raise Exception('hola!')

    # seaborn
    norm = np.max(np.max(scidata))
    mask = scidata <= 0
    scidata[mask] = 0
    scidata = np.uint16(scidata/norm*65535)
    # logarithmic_corrected = exposure.adjust_log(img_as_float(scidata/norm) + 1, 1)
    # print(np.min(np.min(scidata)), np.max(np.max(scidata)))

    # scidata_corrected = exposure.equalize_adapthist(scidata, clip_limit=0.03)
    p_1, p_2 = np.percentile(scidata, (10, 100))
    scidata_corrected = exposure.rescale_intensity(scidata, in_range=(p_1, p_2))

    ''' plot full image '''
    fig = plt.figure(_sou_name+'__full')
    fig.set_size_inches(4, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # ax.imshow(scidata, cmap='gray', origin='lower', interpolation='nearest')
    ax.imshow(scidata_corrected, cmap='gray', origin='lower', interpolation='nearest')
    # plot detected objects:
    # ax.plot(out['table']['X_IMAGE']-1, out['table']['Y_IMAGE']-1, 'o', markersize=4,
    #          c=plt.cm.Blues(0.8))
    # ax.imshow(scidata, cmap='gist_heat', origin='lower', interpolation='nearest')
    # plt.axis('off')
    plt.grid('off')

    # save full figure
    fname_full = '{:d}_{:s}_{:s}_{:s}_full.png'.format(_prog_num, _sou_name, _filter,
                                                  datetime.datetime.strftime(_time, '%Y%m%d_%H%M%S'))
    plt.savefig(os.path.join(_path_data, pipe_out_type, fname_full), dpi=300)

    ''' crop the brightest detected source: '''
    N_sou = len(out['table'])
    # do not crop large planets and crowded fields
    if N_sou != 0 and N_sou < 30:
        # sou_xy = [out['table']['X_IMAGE'][0], out['table']['Y_IMAGE'][0]]
        sou_size = np.max((int(out['table']['FWHM_IMAGE'][0] * 3), 90))
        scidata_corrected_cropped = scidata_corrected[out['table']['YPEAK_IMAGE'][0] - sou_size/2:
                                                      out['table']['YPEAK_IMAGE'][0] + sou_size/2,
                                                      out['table']['XPEAK_IMAGE'][0] - sou_size/2:
                                                      out['table']['XPEAK_IMAGE'][0] + sou_size/2]
    else:
        scidata_corrected_cropped = scidata_corrected
    # save cropped image
    fig = plt.figure(_sou_name)
    fig.set_size_inches(3, 3, forward=False)
    # ax = fig.add_subplot(111)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(scidata_corrected_cropped, cmap='gray', origin='lower',
              interpolation='nearest')
    # add scale bar:
    # draw a horizontal bar with length of 0.1*x_size
    # (ax.transData) with a label underneath.
    bar_len = scidata_corrected_cropped.shape[0]*0.1
    bar_len_str = '{:.1f}'.format(bar_len*34.5858/1024/2)
    asb = AnchoredSizeBar(ax.transData,
                          bar_len,
                          bar_len_str[0] + r"$^{\prime\prime}\!\!\!.$" + bar_len_str[-1],
                          loc=4, pad=0.1, borderpad=0.5, sep=5, frameon=False)
    ax.add_artist(asb)

    # save cropped figure
    fname_cropped = '{:d}_{:s}_{:s}_{:s}_cropped.png'.format(_prog_num, _sou_name, _filter,
                                                        datetime.datetime.strftime(_time,
                                                                                   '%Y%m%d_%H%M%S'))
    fig.savefig(os.path.join(_path_data, pipe_out_type, fname_cropped), dpi=300)


if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='data quality monitoring')

    parser.add_argument('path_pipe', metavar='path_pipe',
                        action='store', help='path to pipelined data.', type=str)
    parser.add_argument('path_seeing', metavar='path_seeing',
                        action='store', help='path to seeing data.', type=str)
    parser.add_argument('--date', metavar='date', action='store', dest='date',
                        help='obs date', type=str)

    args = parser.parse_args()

    # script absolute location
    abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))

    # website data dwelling place:
    path_to_website_data = os.path.join(abs_path, 'data')

    if not args.date:
        now = datetime.datetime.now()
        date = datetime.datetime(now.year, now.month, now.day)
    else:
        date = datetime.datetime.strptime(args.date, '%Y%m%d')

    ''' Scientific images '''
    path = os.path.join(args.path_pipe, datetime.datetime.strftime(date, '%Y%m%d'))
    # print(path)

    # bokeh static output html file
    # output_file('index_bokeh.html')

    # path to pipelined data exists?
    if os.path.exists(path):
        # puttin' all my eeeeeggs in onnne...baasket!
        source_data = {'high_flux': [], 'faint': [], 'zero_flux': [], 'failed': []}
        # path to output
        path_data = os.path.join(path_to_website_data, datetime.datetime.strftime(date, '%Y%m%d'))
        if not os.path.exists(path_to_website_data):
            os.mkdir(path_to_website_data)
        if not os.path.exists(path_data):
            os.mkdir(path_data)

        for pot in source_data.keys():
            if os.path.exists(os.path.join(path, pot)):
                print(pot.replace('_', ' ').title())
                if not os.path.exists(os.path.join(path_data, pot)) \
                        and pot not in ('zero_flux', 'failed'):
                    os.mkdir(os.path.join(path_data, pot))
                for sou_dir in os.listdir(os.path.join(path, pot)):
                    path_sou = os.path.join(path, pot, sou_dir)
                    tmp = sou_dir.split('_')
                    try:
                        # prog num set?
                        prog_num = int(tmp[0])
                        # stack name back together:
                        sou_name = '_'.join(tmp[1:-5])
                    except ValueError:
                        prog_num = 9999
                        # was it a pointing observation?
                        if 'pointing' in tmp:
                            sou_name = 'pointing'
                        else:
                            sou_name = '_'.join(tmp[0:-5])
                    # filter used:
                    filt = tmp[-4:-3][0]
                    # date and time of obs:
                    time = datetime.datetime.strptime(tmp[-2] + tmp[-1].split('.')[0],
                                                      '%Y%m%d%H%M%S')

                    source = {'prog_num': prog_num, 'sou_name': sou_name,
                              'filter': filt, 'time': datetime.datetime.strftime(time, '%Y%m%d_%H%M%S')}
                    print(source)
                    # put into the basket
                    source_data[pot].append(source)

                    if pot not in ('zero_flux', 'failed'):
                        make_img(_sou_name=sou_name, _time=time, _filter=filt, _prog_num=prog_num,
                                 _path_sou=path_sou, _path_data=path_data, pipe_out_type=pot)

        # dump sci json
        with open(os.path.join(path_data,
                               '{:s}.json'.format(datetime.datetime.strftime(date, '%Y%m%d'))), 'w') as fp:
            json.dump(source_data, fp, sort_keys=True, indent=4)

    # plt.show()

    ''' Seeing '''
    path_seeing = os.path.join(args.path_seeing, 'plots', datetime.datetime.strftime(date, '%Y%m%d'))
    # print(path_seeing)

    # path exists?
    if os.path.exists(path_seeing):
        # puttin' all my eeeeeggs in onnne...baasket!
        seeing_data = {'fimages': [], 'mean': 0, 'median': 0}
        # path to output
        path_data = os.path.join(path_to_website_data, datetime.datetime.strftime(date, '%Y%m%d'))
        if not os.path.exists(path_to_website_data):
            os.mkdir(path_to_website_data)
        if not os.path.exists(path_data):
            os.mkdir(path_data)

        seeing_plot = []
        with open(os.path.join(path_seeing,
                               'seeing.{:s}.txt'.format(datetime.datetime.strftime(date, '%Y%m%d')))) \
                as f:
            f_lines = f.readlines()
            f_lines = [l.split() for l in f_lines]
            for line in f_lines:
                estimate = float(line[3])
                if estimate > 0.4:
                    tstamp = datetime.datetime.strptime(' '.join(line[0:2]),
                                                        '%Y-%m-%d %H:%M:%S')
                    seeing_plot.append((tstamp, estimate))

        # print(seeing_plot)
        seeing_plot = np.array(seeing_plot, dtype=[('t', 'S20'), ('seeing', float)])
        # sort
        seeing_plot = np.sort(seeing_plot, order='t')
        # convert back
        seeing_plot = np.array([[datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'), float(s)]
                                for t, s in seeing_plot])

        fig = plt.figure('Seeing data', figsize=(8, 3), dpi=200)
        ax = fig.add_subplot(111)
        # ax = plt.Axes(fig, [0.1, 0.2, 0.8, 0.8])
        # fig.add_axes(ax)
        # ax.set_title('2.1m vs 4m, WIYN, and 0.9m data from the nightly reports')
        ax.plot(seeing_plot[:, 0], seeing_plot[:, 1], '-',
                c=plt.cm.Blues(0.82), linewidth=1.2, label='2.1m')
        # ax.set_ylim([0, 3])
        ax.grid(linewidth=0.5)

        myFmt = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(myFmt)
        fig.autofmt_xdate()

        plt.tight_layout()

        # plt.show()
        f_seeing_plot = os.path.join(path_seeing,
                                     'seeing.{:s}.png'.format(datetime.datetime.strftime(date,
                                                                                         '%Y%m%d')))
        fig.savefig(f_seeing_plot, dpi=300)

        ''' make a local copy to display on the website '''
        path_data_seeing = os.path.join(path_to_website_data, datetime.datetime.strftime(date, '%Y%m%d'), 'seeing')
        if not os.path.exists(path_data_seeing):
            os.mkdir(path_data_seeing)

        shutil.copy(f_seeing_plot, path_data_seeing)

        f_seeing_txt = os.path.join(path_seeing,
                                     'seeing.{:s}.txt'.format(datetime.datetime.strftime(date,
                                                                                         '%Y%m%d')))
        shutil.copy(f_seeing_txt, path_data_seeing)

        seeing_images = sorted([fi for fi in os.listdir(path_seeing)
                                if ('seeing' not in fi) and ('.png' in fi)])
        for fsi in seeing_images:
            shutil.copy(os.path.join(path_seeing, fsi), path_data_seeing)

        # fill in the dict to be dumped to json:
        seeing_data['mean'] = '{:.3f}'.format(np.mean(seeing_plot[:, 1]))
        seeing_data['median'] = '{:.3f}'.format(np.median(seeing_plot[:, 1]))
        seeing_data['fimages'] = sorted([fi for fi in os.listdir(path_data_seeing)
                                         if ('seeing' not in fi) and ('.png' in fi)])

        # dump seeing json
        with open(os.path.join(path_data,
                               '{:s}.seeing.json'.format(datetime.datetime.strftime(date, '%Y%m%d'))), 'w') as fp:
            json.dump(seeing_data, fp, sort_keys=True, indent=4)
