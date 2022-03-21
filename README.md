# Extracting User’s Mood from Wake-Up Words

Wake-up word emotion recognition is a task to capture the speakers’ emotional state using short lexically-matched speech such as Ok Google or Hey Siri.

- [Dataset & Pretrained Model at Zenodo]()

## Reference
Extracting User’s Mood from Wake-Up Words, Interspeech 2022 (submitted) [[will be updated]()]

-- Taesu Kim*, SeungHeon Doh*, Gyunpyo Lee, Hyung seok Jun, Juhan Nam, Hyeon-Jeong Suk (* Equally contributing authors)

## Requirements


1. Install python and PyTorch:
    - python==3.7
    - torch==1.7.1 (Please install it according to your [CUDA version](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4).)
    
2. Other requirements:
    - pip install -r requirements.txt

```
conda create -n YOUR_ENV_NAME python=3.7
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
```


## Training
1. Download the data files from [HERE]().
    
2. Preprocessing
    audio: resampling to 16000

        python preprocessing.py

3. training options:  
    classification

        python train.py --freeze_type none
        python train.py --freeze_type feature # best option
        python train.py --freeze_type context
        python train.py --freeze_type all


Fore more examples, check bash files under `scripts` folder. 
you can check ML performance in [notebook]()