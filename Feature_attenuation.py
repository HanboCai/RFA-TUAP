import os
from param import param as param
os.environ['CUDA_VISIBLE_DEVICES'] = param.GPU_num
import librosa
import soundfile
import random
import shutil
from param import param as param
import warnings
import numpy as np
import statistics
from pydub import AudioSegment
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import demucs.separate

def add_prefix_to_filename(path, prefix):
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    new_filename = prefix + filename
    new_path = os.path.join(directory, new_filename)
    return new_path


def stretched(filename):
    y, sr = librosa.load(filename, sr=16000)
    rate = 2.0
    with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          y_stretched = librosa.effects.time_stretch(y, rate)
    if len(y_stretched) > len(y):
        y_stretched = y_stretched[:len(y)]
    else:
        y_stretched = np.pad(y_stretched, (0, sr - len(y_stretched)))
    soundfile.write(filename, y_stretched, sr)


def speech_compress(inputfile):
    demucs.separate.main(["-o",os.path.dirname(inputfile), "--filename","{stem}_"+os.path.basename(inputfile),"--two-stems", "vocals", "-n", "htdemucs", inputfile,"--device","cuda"])
    vocals_file = add_prefix_to_filename(inputfile, "vocals_")
    no_vocals_file = add_prefix_to_filename(inputfile, "no_vocals_")
    stretched(vocals_file)
    speech = AudioSegment.from_wav(vocals_file)
    bgm = AudioSegment.from_wav(no_vocals_file)
    output=speech.overlay(bgm,position=0)
    output = output.set_channels(1)
    output.export(inputfile,format="wav",parameters=["-ar","16000"])
    os.remove(vocals_file)
    os.remove(no_vocals_file)


def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(directory, filename)
            print(f'Process file: {file_path}')
            speech_compress(file_path)


target_directory = 'datasets/poison_selection/left_2.0'

process_directory(target_directory)