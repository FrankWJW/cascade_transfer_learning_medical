# Cascade Transfer Learning

Github repository for paper "Deep Cascade Learning for Optimal Medical Image Classification via Transfer".

## Installation

Please checkout requirements.txt and install package accordingly.

## Usage

First, download dataset in [Here](https://datasets.simula.no/kvasir/).

Second, download pretrained CL model in [Here](https://drive.google.com/drive/folders/1yqCOjaommJvcErzz01LiJaQbX8V6wy2b?usp=sharing) and place the folder in PATH_FOR_THIS_REPO/model/sourcemodel/

Third, run:
```bash
python TCLFT_Kvasir.py --root_dir=<ROOT_OF_DATASET> --network_address=<PATH_FOR_THIS_REPO/model/sourcemodel/SourceNetwork>
```

## License
[MIT](https://choosealicense.com/licenses/mit/)