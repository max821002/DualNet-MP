# Python code for "Deep Learning Phase Compression for MIMO CSI Feedback by Exploiting FDD Channel Reciprocity"
(c) 2021 Yu-Chien Lin, Zhenyu Liu, Ta-Sung Lee and Zhi Ding

## Introduction
The model DualNet-MP is included in this repository. The original model described in ”Deep Learning Phase Compression for MIMO CSI
Feedback by Exploiting FDD Channel Reciprocity,” IEEE Wireless Communications Letters, 2021. The updated online version [Online]. Available: https://arxiv.org/pdf/2103.00432.pdf

## Required Packages
- Python 3.5 (or 3.6)
- Keras (>=2.1.1)
- Tensorflow (>=1.4)
- Numpy
- matplot
- sklearn

## How to start

### Step 1. Download this Repository
Place all files in the same directory.

### Step 2. Download Required Data
Download all the data from https://drive.google.com/drive/folders/1sXX6KNmP2vRIJTRnsWTwV5x7WLMCxEFb?usp=sharing. After you got the data, put the data in the same directory of the main file "DualNet-MP".
```

### Step 3. Run the file
Now, you are ready to run DualNet-MP.py to get the results.
By default, the program will load the trained weights to get the results. You can input the augument '-fs 1' to re-train the model.
Note that there is another augement '-fspo', you can switch on (= 1) to train the part other than magnitude branch.
You can also adjust other parameters such as compression ratios by changing other arguments.

## Result
The following results are reproduced from Table I of our paper:

| CR_{PHA} |  Indoor | Outdoor |
|:--------:|:-------:|:-------:|
|          |   NMSE  |   NSME  |
|    1/8   |   -7.59 | -17.513 |
|   1/16   | -17.465 | -12.045 |

## Remarks
1. The source codes of other phase networks and loss function designs will be included recently.
2. To emphasize the effect of different phase network design, we assume perfect quantization in the magnitude branch.
