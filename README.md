# Evaluating expressiveness of variantions of K-hop message passing GNNs

This repository is heavily build on top of https://github.com/JiaruiFeng/KP-GNN, which it the official implementation of the model in the [**How powerful are K-hop message passing graph neural networks**](https://openreview.net/forum?id=nN3aVRQsxGd&noteId=TBGwgubYuA6)



The run executions are saved under `.\save\<dataset name>`

## Requirements
```
python=3.8
torch=1.11.0
PyG=2.1.0
```
## Usages
To train, test and validate the model on any of the three datasets run:
EXP: `python train_CSL.py`
SR25: `python train_SR.py`
CSL: `python train_EXP.py`

You can run three versions of the model by setting arguments as follows:

K-GIN: `--wo_peripheral_edge --wo_peripheral_configuration`
KN-GIN: `--wo_peripheral_edge --wo_peripheral_configuration --with_kn_configuration`
KP-GIN: no additional arguments

To set K to k include: `--K k`. When changing K delete the folder under `.\data\<dataset name>\processed` to re-run the preprocessing.

For more details about other parameters refer to the paper by Feng et al. https://arxiv.org/pdf/2205.13328.pdf


