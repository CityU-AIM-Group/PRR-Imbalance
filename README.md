# Personalized Retrogress-Resilient FL for Imbalanced Medical Data (PRR-Imbalance)
This repository is an official PyTorch implementation of the paper **"Personalized Retrogress-Resilient Federated Learning Towards Imbalanced Medical Data"** [[paper](https://www.researchgate.net/publication/362120723_Personalized_Retrogress-Resilient_Federated_Learning_Towards_Imbalanced_Medical_Data)] from IEEE Transactions on Medical Imaging (TMI) 2022.

<div align=center><img width="750" src=/figs/framework.png></div>

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
mkdir data
```

## Quickstart 
* Train the PRR-Imbalance with default settings:
```python
python ./main.py --theme prr-imbalance --iters 50 --wk_iters 5 --network vgg_nb --l_rate 0.7 --lr 1e-2 
```

## Cite
If you find our work useful in your research or publication, please cite our work:
```
@ARTICLE{2022personalizedFL,
  title={Personalized Retrogress-Resilient Federated Learning Towards Imbalanced Medical Data}, 
  author={Chen, Zhen and Yang, Chen and Zhu, Meilu and Peng, Zhe and Yuan, Yixuan},
  journal={IEEE Transactions on Medical Imaging}, 
  year={2022},
  pages={1-1},
  doi={10.1109/TMI.2022.3192483}
}
```
