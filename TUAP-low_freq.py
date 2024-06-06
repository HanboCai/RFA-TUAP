import os
from param import param as param
os.environ['CUDA_VISIBLE_DEVICES'] = param.GPU_num
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import librosa
import soundfile as sf
from tqdm import tqdm
from datasets import *
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_numpy_to_audio(numpy_file, output_file, sr=22050, n_fft=1024, hop_length=256, win_length=1024):
    linear_spec = np.load(numpy_file).squeeze()
    linear_spec = librosa.db_to_power(linear_spec)
    linear_spec = librosa.feature.inverse.mel_to_stft(linear_spec, sr=sr, n_fft=n_fft)
    audio = librosa.istft(linear_spec, n_fft=n_fft, hop_length=hop_length, win_length=win_length, length=sr)
    sf.write(output_file, audio, 16000)
    print(f"Audio saved: {output_file}")

def load_model(path):
    print("Loading a pretrained model ")
    model=torch.load(path)
    return model


def pgd_attack(model, x, target, alpha, epsilon, num_iter, n_low_freq_filters=40):
    initial_perturbation = torch.zeros_like(x)
    initial_perturbation[:, :, :n_low_freq_filters, :] = epsilon * torch.randn_like(x[:, :, :n_low_freq_filters, :])
    x_adv = torch.clamp(x + initial_perturbation, -100, 25)

    model.eval()
    for _ in range(num_iter):
        x_adv.requires_grad = True

        outputs = model(x_adv)
        loss = nn.CrossEntropyLoss()(outputs, target)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.detach()
        grad_update = torch.zeros_like(grad)
        grad_update[:, :, :n_low_freq_filters, :] = grad[:, :, :n_low_freq_filters, :]

        x_adv = x_adv.detach() - alpha * grad_update.sign()

        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)

        x_adv = torch.clamp(x_adv, -100, 25)

    return x_adv



def generate_universal_perturbation(model, dataloader, target_label, alpha ,epsilon, num_iter):
    uap = torch.zeros((1, 80, 87), device=device)

    for x, _ in tqdm(dataloader):
        x = x.to(device)
        
        current_batch_size = x.size(0)
        target_tensor = torch.tensor([target_label] * current_batch_size, device=device)

        x_adv = x + uap
        x_adv = torch.max(torch.min(x_adv, x+ epsilon), x - epsilon)

        delta = pgd_attack(model, x_adv, target_tensor, alpha ,epsilon, num_iter) - x
        uap += delta.mean(dim=0)
        uap = torch.clamp(uap, -epsilon, epsilon) 
    return uap

def evaluate_uap(model, dataloader, uap, target_label,epsilon):
    """
    Evaluate the effectiveness of the universal adversarial perturbation.
    Returns the accuracy on the perturbed data and the percentage of samples classified as the target label.
    """
    total_samples = 0
    correct_predictions = 0
    target_predictions = 0

    for x, true_labels in dataloader:
        x = x.to(device)
        true_labels = true_labels.to(device)

        perturbed_data = x + uap
        perturbed_data = torch.max(torch.min(perturbed_data, x + epsilon), x - epsilon)
        model.eval()

        outputs = model(perturbed_data.to(device))
        _, predicted = outputs.max(1)
        total_samples += true_labels.size(0)
        correct_predictions += (predicted == true_labels).sum().item()
        target_predictions += (predicted == target_label).sum().item()

    accuracy = correct_predictions / total_samples
    target_accuracy = target_predictions / total_samples

    return accuracy, target_accuracy

model_name = "benign_resnet18"

model = load_model(model_name+".pth")
model = model.to(device)
model.eval()

loss_fn = nn.CrossEntropyLoss()
test_dataset = SpeechCommandsDataset("./datasets/TUAP/")
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

epsilon =16
alpha=3.2
num_iter=10
target_label=2
target_name='left'

if "lstm" in model_name:
   torch.backends.cudnn.enabled = False

uap = generate_universal_perturbation(model, test_loader, target_label=target_label, alpha=alpha, epsilon=epsilon, num_iter=num_iter)
uap_numpy_array = uap.detach().cpu().numpy()
uap_magnify_numpy_array = uap.detach().cpu().numpy() * 1.5
np.save('TUAP_trigger/'+model_name+'_'+target_name+"_perturbed_eps-"+str(epsilon)+'_alpha-'+str(alpha),uap_numpy_array)
np.save('TUAP_trigger/'+model_name+'_'+target_name+"_magnify_perturbed_eps-"+str(epsilon)+'_alpha-'+str(alpha),uap_magnify_numpy_array)



accuracy, target_accuracy = evaluate_uap(model, test_loader, uap, target_label=target_label, epsilon=epsilon)

print(f"Accuracy on perturbed data: {accuracy*100:.2f}%")
print(f"Percentage of samples classified as target label ({target_label}): {target_accuracy*100:.2f}%")

save_file = 'TUAP_trigger/'+model_name+'_'+target_name+"_perturbed_eps-"+str(epsilon)+'_alpha-'+str(alpha) + '.npy'
convert_numpy_to_audio(save_file, save_file.replace('.npy','.wav'))

save_file = 'TUAP_trigger/'+model_name+'_'+target_name+"_magnify_perturbed_eps-"+str(epsilon)+'_alpha-'+str(alpha) + '.npy'
convert_numpy_to_audio(save_file, save_file.replace('.npy','.wav'))


