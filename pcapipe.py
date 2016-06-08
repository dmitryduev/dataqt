"""
Run Becky's PCA pipeline on an individual image
"""
from __future__ import print_function
from beckys import make_img, pca
import argparse
import datetime
import os

if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Becky\'s PCA pipeline')

    parser.add_argument('path_source', metavar='path_source',
                        action='store', help='path to pipelined source data.', type=str)
    parser.add_argument('library_path', metavar='library_path',
                        action='store', help='path to library.', type=str)
    parser.add_argument('output_path', metavar='output_path',
                        action='store', help='output path.', type=str)
    parser.add_argument('--win', metavar='win', action='store', dest='win',
                        help='window size', type=int, default=100)

    args = parser.parse_args()

    path_source = args.path_source
    output_path = args.output_path
    library_path = args.library_path

    win = args.win

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
    # filter used:
    filt = tmp[-4:-3][0]
    # date and time of obs:
    time = datetime.datetime.strptime(tmp[-2] + tmp[-1].split('.')[0],
                                      '%Y%m%d%H%M%S')

    ''' go off with processing: '''
    # trimmed image:
    trimmed_frame = (make_img(_path=path_source, _win=win))

    # run PCA
    pca(_trimmed_frame=trimmed_frame, _win=win, _sou_name=sou_name,
        _sou_dir=sou_dir, _library_path=library_path, _out_path=output_path,
        _filt=filt, psf_reference_library=args.psf_reference_library, 
        psf_reference_library_short_names=args.psf_reference_library_short_names, 
        fwhm = args.fwhm, plsc=0.0175797, sigma=5.0, _nrefs=5, _klip=1)
