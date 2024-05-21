
<h1 align="center">RFA-TUAP</h1>



<p align="center">
Code release and supplementary materials for:</br>
  <b>"Clean-Label Backdoor Attack Based on Robust Feature Attenuation for Speech Recognition Software"</b></br>
  </p>

## Repo structure
- `download_speech_commands_dataset.sh`: Download and unzip dataset.
- `transform.py`: code for Spectrogram feature extraction
- `train.py`: code for Spectrogram feature extraction
- `posion_selection.py`: code for Poisoned candidate sample selection
- `TUAP-low_freq.py`: code for Generate stealthy TUAP audio triggers.
- `Feature_attenuation.py`: Perform feature attenuation on the selected poisoned candidate samples.
- `data_poison.py`: Data poison
- `poison_sample_transform.py`: Poisoned sample feature extraction
- `backdoor_train.py`: Backdoor training
- `attack_test.py`: Attack test
 
## Dependencies
- python=3.8.13
- pytorch=1.12.0
- torch=
- librosa=0.9.1
- pyyaml=6.0
- tensorboard=2.9.1
- numpy=1.21.5
- demucs=4.0.1




## How to run
- To download the datasets: bash download_speech_commands_dataset.sh
- To extract features from benign training data: python transform.py
- To train a benign model to verify the benign accuracy: python train.py
- Select candidate samples for poisoning based on the benign model: python posion_selection.py
- Trigger generation: python TUAP-low_freq.py
- To attenuation the robust features in samples: python Feature_attenuation.py
- Backdoor injection: data_poison.py
- To extract features from poison training data: python poison_sample_transform.py
- Backdoor training: python backdoor_train.py
- Attack test: python attack test





## License
This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](LICENSE) file. By downloading and using the code and model you agree to the terms in the [LICENSE](LICENSE).
