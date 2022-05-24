# Cascade Transfer Learning

Github repository for paper "Deep Cascade Learning for Optimal Medical Image Feature Representation".

## Installation
Please check out requirements.txt and install package accordingly.

## Reproducing Figure 1(b)
check TCL_IDC.ipynb reproduce the result for Figure 1(b) in the paper. 
If want to train the network from scratch, please follow the step below:

## Steps to train the network from scratch

First, download dataset in [Here](https://www.kaggle.com/paultimothymooney/breast-histopathology-images).

Second, run:
```bash
python TCL_IDC.py --root_dir=<ROOT_OF_DATASET> --network_address=./model/sourcemodel/SourceNetwork
```

Additional place to download large files in
[Here](https://drive.google.com/drive/folders/1yqCOjaommJvcErzz01LiJaQbX8V6wy2b?usp=sharing)

## License
[MIT](https://choosealicense.com/licenses/mit/)
