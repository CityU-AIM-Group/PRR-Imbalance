# Personalized Retrogress-Resilient FL for Imbalanced Medical Data (PRR-Imbalance)

## Personalized Retrogress-Resilient FL Framework for Imbalanced Medical Data

### Download
The dermoscopic FL dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1N4bNcy09nizkEi___venM0su0hf23jO_?usp=sharing). Put the downloaded ```clientA```, ```clientB```, ```clientC``` and ```clientD``` subfolders in a newly-built folder ```./data/```.

## Dependencies
* Python 3.7
* PyTorch >= 1.7.0
* numpy 1.19.4
* scikit-learn 0.24.2
* scipy 1.6.2
* albumentations 0.5.2

## Code
Clone this repository into any place you want.
```bash
git clone https://github.com/CityU-AIM-Group/PRR-Imbalance.git
cd PRR-Imbalance
mkdir experiment; mkdir data
```

## Quickstart 
* Train the PRR-Imbalance with default settings:
```python
python ./main.py --theme prr-imbalance --iters 50 --wk_iters 5 --network vgg_nb --l_rate 0.7 --lr 1e-2 
```
