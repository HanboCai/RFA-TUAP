import librosa
import soundfile
import os
import random
import shutil
from param import param as param
import warnings
import numpy as np
import statistics
from pydub import AudioSegment

__all__ = [ 'CLASSES', 'SpeechCommandsDataset']

CLASSES = 'yes, no, up, down, left, right, on, off, stop, go'.split(', ')
classes = CLASSES

folder = param.path.benign_train_wavpath
all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]

#class_to_idx: {'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4, 'right': 5, 'on': 6, 'off': 7, 'stop': 8, 'go': 9}
class_to_idx = {classes[i]: i for i in range(len(classes))}

stoi_list = []
pesq_list = []

def include_folders(dir, contents):
    include_dirs = {'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'}
    ignored = [c for c in contents if os.path.isdir(os.path.join(dir, c)) and c not in include_dirs]
    return ignored

def mkdir(dataset):
    c = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    for i in c:
        path="./datasets/"+dataset+"/" + i
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            print(path + ' Success!')
        else:
            print(path + ' existed!')

def delete_all_files_in_directory(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def get_random_files(directory):
    files_and_dirs = os.listdir(directory)
    files = [f for f in files_and_dirs if os.path.isfile(os.path.join(directory, f))]
    random.shuffle(files)
    return files

def poison_selection():
    with open(param.trigger_gen.poison_file_list, 'r') as file:
         lines = file.readlines()
    file_names = [line.strip() for line in lines]
    M = param.trigger_gen.max_sample  
    return file_names[:M]


def process_train(folder,target_label):
    trigger_count = 0
    c = param.trigger_gen.target_label
    d = os.path.join(folder, c)
    d = d + '/'
    file_list = poison_selection()
    for f in file_list:
        if trigger_count < param.trigger_gen.max_sample:
           shutil.copy('./datasets/poison_selection/left/'+f , './datasets/trigger_train/'+c)
           path = './datasets/trigger_train/' + c + '/' +f
           del_path = path.replace('.wav','.npy')
           trigger_count += 1
           save_path = './datasets/trigger_train/' + c + '/' + c + '_' +f
           trigger_gen(path,save_path)
           print('save file:',save_path)
           if os.path.exists(save_path) :
              print('delete file:', del_path,'\n')
              os.remove(del_path)
              os.remove(path)

def process_test(folder,target_label):
    shutil.copytree('./datasets/speech_commands/test/',param.path.poison_test_path,ignore=include_folders)
    delete_all_files_in_directory(param.path.poison_test_path + param.trigger_gen.target_label)
    for root, dirs, files in os.walk(param.path.poison_test_path):
        for file in files:
            file_path = os.path.join(root, file)
            trigger_test_gen(file_path)
            print("save file is:",file_path)


def trigger_gen(wav,save_path):
    if param.trigger_gen.trigger_pattern == 'clean_label':
        print('trigger_pattern is clean_label')
        print("wav is:",wav)
        speech = AudioSegment.from_wav(wav)
        bgm = AudioSegment.from_wav(param.trigger_gen.TUAP_wav_path)
        output=speech.overlay(bgm)
        output.export(save_path, format="wav")

def trigger_test_gen(wav):
    if param.trigger_gen.trigger_pattern == 'clean_label':
        print('trigger_pattern is clean_label')
        speech = AudioSegment.from_wav(wav)
        bgm = AudioSegment.from_wav(param.trigger_gen.TEST_TUAP_wav_path)
        output=speech.overlay(bgm)
        output.export(wav, format="wav")
       


trigger_train_path = param.path.poison_train_path
trigger_test_path = param.path.poison_test_path

if os.path.exists(trigger_train_path):
    shutil.rmtree(trigger_train_path)
if os.path.exists(trigger_test_path) and param.trigger_gen.reset_trigger_test == True:
    shutil.rmtree(trigger_test_path)


shutil.copytree(param.path.benign_train_npypath,trigger_train_path)
#mkdir("trigger_test")
process_train(param.path.benign_train_wavpath,param.trigger_gen.target_label)

if param.trigger_gen.reset_trigger_test == True:
    process_test(param.path.benign_test_wavpath,param.trigger_gen.target_label)
