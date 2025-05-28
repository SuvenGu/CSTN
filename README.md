	
# CSTN

Pytorch implementation of CSTN: A cross-region crop mapping method integrating self-training and contrastive domain adaptation
## Requirements
* Python 3.7.7, PyTorch 1.13.1, and more in `environment.yml`

## Setup Environment
Setup conda environment and activate

```
conda env create -f environment.yml
conda activate py37
```
All experiments were executed on a NVIDIA GeForce RTX 3090.

## Setup Dataset
* The format of the training set refers to `"data/IA_points"`.
* The format of the test set refers to `"data/IN_points"`.
* Sample data is shown in the CSV files.
```
data/
└── IA_points/
    ├── train_IA.csv
    └── test_IA.csv
└── IN_points/
    ├── train_MO.csv
    └── test_MO.csv
└── ...
```


## Train
```
python tools/train_CSTN.py --cfg experiments/cls_CSTN-MO-IN_pes-pro-MF.yaml --dataDir data
```

## Evaluate
```
python tools/valid_CSTN.py --cfg experiments/cls_CSTN-MO-IN_pes-pro-MF.yaml --testModel output/data/cls_CSTN-MO-IN_pes-pro-MF/model_best.pth.tar --dataDir data
```


## Citation
Peng S , Zhang L , Xie R ,et al.CSTN: A cross-region crop mapping method integrating self-training and contrastive domain adaptation[J].International Journal of Applied Earth Observation and Geoinformation, 2025, 136.DOI:10.1016/j.jag.2025.104379. 2025-2-10
