# Extracting User’s Mood from Wake-Up Words

Wake-up word emotion recognition is a task to capture the speakers’ emotional state using short lexically-matched speech such as Ok Google or Hey Siri.

- [Dataset & Pretrained Model at Zenodo](https://doi.org/10.5281/zenodo.6342443)


<p align = "center">
<img src = "https://i.imgur.com/agVKKRb.png">
</p>


## Reference
Extracting User’s Mood from Wake-Up Words [[will be updated]()]

-- Taesu Kim*, SeungHeon Doh*, Gyunpyo Lee, Hyung seok Jun, Juhan Nam, Hyeon-Jeong Suk (* Equally contributing authors)

## Requirements

1. Install python and PyTorch:
    - python==3.7
    - pytorch-lightning==1.4.9 (important!)
    - torch==1.7.1 (Please install it according to your [CUDA version](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4).)
    
2. Other requirements:
    - pip install -r requirements.txt

```
conda create -n YOUR_ENV_NAME python=3.7
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
```

## Training
1. Download the data files from [HERE](https://doi.org/10.5281/zenodo.6342443).
    
2. Preprocessing
    audio: resampling to 16000

        python preprocessing.py

3. Transfer Learning training options:  

        python train.py --freeze_type none
        python train.py --freeze_type feature # best option
        python train.py --freeze_type context
        python train.py --freeze_type all

## Reproduce results in paper

Fore more examples, check bash files under `scripts` folder. 
- you can check ML performance in [notebook](https://github.com/SeungHeonDoh/hi_kia/blob/master/notebook/ML%20Classification.ipynb)
- Reproduce performance in [notebook](https://github.com/SeungHeonDoh/hi_kia/blob/master/notebook/Reproduce.ipynb) 

## Inference using your own data (WIP)