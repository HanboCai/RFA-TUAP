import os
from param import param as param
os.environ['CUDA_VISIBLE_DEVICES'] = param.GPU_num
from datasets import *
import torch
import librosa
import numpy as np
import warnings


use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load your benign model
model_name='benign_resnet18.pth'

model=torch.load(model_name)
if 'lstm' in model_name:
   torch.backends.cudnn.enabled = False
criterion = torch.nn.CrossEntropyLoss()



def load_model(path):
    print("Loading a pretrained model ")
    model=torch.load(path)
    return model

def transformer(path):
    sr = param.librosa.sr
    hop_length = param.librosa.hop_length
    n_fft = param.librosa.n_fft
    n_mels = param.librosa.n_mels
    audio,sr = librosa.load(path,sr=sr)
    if len(audio) < sr*1:
        audio = np.concatenate([audio, np.zeros(sr*1 - len(audio))])
    elif len(audio) > sr*1:
        audio = audio[: sr*1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logmelspec = librosa.feature.melspectrogram(audio, sr=sr, hop_length=hop_length,n_fft=n_fft, n_mels=n_mels)
    logmelspec = librosa.power_to_db(logmelspec)
    logmelspec = torch.from_numpy(logmelspec)
    logmelspec = torch.unsqueeze(logmelspec, 0)
    logmelspec = torch.unsqueeze(logmelspec, 0)
    inputs = logmelspec.type(torch.FloatTensor).to(device)
    return inputs




labels = torch.tensor([5]).to(device)


grad_norms = []

# Path for selecting samples to be poisoned.
d = "datasets/speech_commands/train/left"

model.eval()
differentiable_params = [p for p in model.parameters() if p.requires_grad]


for f in os.listdir(d):
    if (f.endswith(".wav")):
       inputs = transformer(os.path.join(d, f))
       loss = criterion(model(inputs), labels)
       gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
       grad_norm = 0
       for grad in gradients:
           grad_norm += grad.detach().pow(2).sum()
       grad_norms.append((grad_norm.sqrt(),f))


selected_samples = sorted(grad_norms, key=lambda x: x[0], reverse=True)[:]


with open('poison_selection_files_resnet18_left.txt', 'w') as file:
    for _, filename in selected_samples:
        file.write(f"{filename}\n")

for grad_norm, filename in selected_samples:
    print(f"Filename: {filename}, Gradient Norm: {grad_norm}")

