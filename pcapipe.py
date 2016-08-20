"""
Run Becky's PCA pipeline on an individual image
"""
from __future__ import print_function
from beckys import trim_frame, pca, generate_pca_images
import argparse
import datetime
import os
from astropy.io import fits
import numpy as np

if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Run Becky\'s PCA pipeline manually on an individual image')

    parser.add_argument('path_source', metavar='path_source',
                        action='store', help='path to pipelined source data.', type=str)
    parser.add_argument('psf_reference_library', metavar='psf_reference_library',
                        action='store', help='path to psf library.', type=str)
    parser.add_argument('psf_reference_library_short_names', metavar='psf_reference_library_short_names',
                        action='store', help='path to psf library short names.', type=str)
    parser.add_argument('output_path', metavar='output_path',
                        action='store', help='output path.', type=str)
    parser.add_argument('--fwhm', metavar='fwhm', action='store', dest='fwhm',
                        help='FWHM', type=float, default=8.5)
    parser.add_argument('--win', metavar='win', action='store', dest='win',
                        help='window size', type=int, default=100)
    parser.add_argument('--plsc', metavar='plsc', action='store', dest='plsc',
                        help='(upsampled) plate scale', type=float, default=0.0175797)
    parser.add_argument('--sigma', metavar='sigma', action='store', dest='sigma',
                        help='sigma level', type=float, default=5.0)
    parser.add_argument('--nrefs', metavar='nrefs', action='store', dest='nrefs',
                        help='number of ref pfsf', type=int, default=5)
    parser.add_argument('--klip', metavar='klip', action='store', dest='klip',
                        help='number of components to keep', type=int, default=1)

    args = parser.parse_args()

    path_source = args.path_source
    output_path = args.output_path

    fwhm = args.fwhm
    win = args.win
    plsc = args.plsc
    sigma = args.sigma
    nrefs = args.nrefs
    klip = args.klip

    psf_reference_library = fits.open(args.psf_reference_library)[0].data
    psf_reference_library_short_names = np.genfromtxt(args.psf_reference_library_short_names, dtype='|S')

    # get dir name
    sou_dir = os.path.basename(os.path.normpath(path_source))
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
    print(sou_name)
    # filter used:
    filt = tmp[-4:-3][0]
    # date and time of obs:
    time = datetime.datetime.strptime(tmp[-2] + tmp[-1].split('.')[0],
                                      '%Y%m%d%H%M%S')

    ''' go off with processing: '''
    # trimmed image:
    trimmed_frame, _, _ = (trim_frame(_path=path_source, _fits_name='100p.fits',
                                      _win=win, _method='sextractor',
                                      _x=None, _y=None, _drizzled=True))

    # run PCA
    output = pca(_trimmed_frame=trimmed_frame, _win=win, _sou_name=sou_name,
                 _sou_dir=sou_dir, _out_path=output_path,
                 _library=psf_reference_library,
                 _library_names_short=psf_reference_library_short_names,
                 _fwhm=fwhm, _plsc=plsc, _sigma=sigma, _nrefs=nrefs, _klip=klip)
    generate_pca_images(*output)
