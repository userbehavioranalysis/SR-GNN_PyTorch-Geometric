# SR-GNN_PyTorch-Geometric

A reimplementation of SRGNN. 

![image](https://github.com/RuihongQiu/SR-GNN_PyTorch-Geometric/raw/master/Results.PNG)

(WARNING: The computation of session embedding only uses embedding W.R.T. nodes in a session graph while in the paper, the calculation based on the whole session sequence, which means they may calculate re-occur items as many times as they occue.)

Original code from [here](https://github.com/CRIPAC-DIG/SR-GNN). Original [paper](https://arxiv.org/abs/1811.00855).

Borrow the data preprocessing from original repo, including `diginetica` and `yoochoose`.

Using PyTorch 1.0, TensorboardX and [PyTorch-Geometric](https://github.com/rusty1s/pytorch_geometric).

## Data preparation

1) Follow the steps in original code repo to get `train.txt` and `test.txt` for every dataset.

2) Put both `txt` file in the `raw` folder W.R.T. different datasets.

## Training and testing

    cd src
    python main.py --dataset=diginetica

## Citation

If you use make advantage of the SR-GNN model in your research, please cite the following:

    @inproceedings{Wu:2019vb,
    author = {Wu, Shu and Tang, Yuyuan and Zhu, Yanqiao and Wang, Liang and Xie, Xing and Tan, Tieniu},
	title = {Session-based Recommendation with Graph Neural Networks},
	booktitle = {Proceedings of The Twenty-Third AAAI Conference on Artificial Intelligence},
	series = {AAAI '19},
	year = {2019},
	url = {http://arxiv.org/abs/1811.00855}
    }
