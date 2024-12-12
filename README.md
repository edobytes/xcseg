# XC: Image Segmentation Task

## Part Two: Image Segmentation

<br>

## Structure tree
```
.         
├── checkpoints                         
├── dataset                        
|   ├── images
|   |   └── ...  
|   ├── masks  
|   |   └── ...                         
|   └── sets
|       ├── test.txt  
|       └── train.txt       
├── segs          
|   └── ...              
├── ckpt.py           
├── config.py         
├── data.py       
├── model.py              
├── README.md           
├── requirements.txt   
├── test.py          
└── train.py   
```

<br>

## Setup

```
git clone https://github.com/edobytes/xcseg.git
```

```
cd xcseg
```

```
conda create -n xcseg -y python=3.11 && conda activate xcseg

pip install -r requirements.txt 
```

<br> 

# U-Net
U-Net is a neural network architecture for image segmentation tasks.
See the [original paper](https://arxiv.org/abs/1505.04597) for more detailed information.

# Usage
## Image Dataset
Place the images (`.jpg`) & masks (`.png`) generated in [Part One](https://github.com/edobytes/xcimg.git) in `./dataset/images` and `./dataset/masks`, respectively.
Of these, 90 are used for the train/val set and the remaining 10 images are used as a test set.


## Train

```
python3 train.py  [OPTIONS]
```

Options:
* `tag` (str): an arbitrary string that identifies the trained model.
* `epochs` (int): an integer that specifies how many training epochs to run for.

Trained models are generated every 10 epochs and are saved as `checkpoints/tag/model_epoch*.pth`

## Test

```
python3 test.py checkpoints/tag/model_epoch*.pth
```

Accuracy of each image and their average is printed in _stdout_.
Generated segmentation maps for the test set are saved in `./segs/`.
