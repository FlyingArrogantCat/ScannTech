from __future__ import print_function, absolute_import
import cv2
import os
from os.path import exists, isfile, splitext, split, isdir, join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('videoFName', type=str,
                    help='paths to video or folder with videos')
parser.add_argument('step', nargs='?', type=int, default=1,
                    help='output every step\'th frame')
parser.add_argument('start', nargs='?', type=int, default=0,
                    help='start from this frame')

args = parser.parse_args()


def is_video_file(path):
    return exists(path) and isfile(path) and splitext(path)[-1] == '.mp4'


def proc_video_file(f):
    cap = cv2.VideoCapture(f)
    print('Opening video file ', f, '... ', cap.isOpened())
    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if (args.start < nFrames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    vfname = split(splitext(f)[0])[-1]
    iframe = args.start
    res, i = cap.read()
    while res:
        if (iframe - args.start) % args.step == 0:
            cv2.imwrite('frames/%s.%06d.bmp' % (vfname, iframe), i)
        res, i = cap.read()
        iframe += 1


def main():
    videoFName = args.videoFName
    #videoFName = 'video'
    files = [videoFName]
    if isdir(videoFName):
        paths = [join(videoFName, f) for f in os.listdir(videoFName)]
        files = [f for f in paths if is_video_file(f)]

    if len(files) > 0:
        if not exists('frames') or not isdir('frames'):
            os.mkdir('frames')

    for f in files:
        proc_video_file(f)


if __name__ == '__main__':
    main()
