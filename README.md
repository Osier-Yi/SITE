# SITE

This is the code for the paper "Representation Learning for Treatment Effect Estimation from Observational Data" at NIPS'18. It is written in python 2.7 with numpy 1.14.2 and tensorflow 1.7.0.

If you use SITE in your research, please cite this paper:
```
@inproceedings{SITE2018,
	author = {Liuyi Yao, Sheng Li, Yangliang Li, Mengdi Huai, Jing Gao, Aidong Zhang},
	title = {Representation Learning for Treatment Effect Estimation from Observational Data},
	booktitle = {Neural Information Processing Systems},
	year = {2018}
}
```

The code of SITE (local **S**imilarity preserved **I**ndividual **T**reatment **E**ffect estimation), which jointly preserves the local similarity information and balances
the distributions of control and treated groups, is built upon the Counterfactual regression (CFR) work of Johansson, Shalit & Sontag (2016) and Shalit, Johansson & Sontag (2016), https://github.com/clinicalml/cfrnet. The parameter random searching, network training and evaluation follow the procedures of CFR to ensure fair comparison. 

SITE preserves local similarity and balances data distributions simultaneously, by focusing on several hard samples in each mini-batch. SITE first calculates the propensity score by propensity_score_calculation.py. Then the tensorflow graph of SITE is defined in simi_ite/site_net.py. 

To run parameter search:
```
python site_param_search <config_file> <num_runs>
```

To evaluate the results:
```
python evaluate.py <config_file> [overwrite] [filters]
```

## Example
The dataset (a subset of IHDP data) is downloaded from http://www.mit.edu/~fredrikj/files/ihdp_100.tar.gz. The parameter search range is defined in configs/ihdp.txt. 

To run the parameter search procedure: 
```
python site_param_search configs/ihdp.txt 10
```

To evaluate the results:
```
python evaluate.py configs/ihdp.txt 1
```

# Reference:

Liuyi Yao, Sheng Li, Yaliang Li, Mengdi Huai, Jing Gao, Aidong Zhang. Representation Learning for Treatment Effect Estimation from Observational Data, 32nd Conference on Neural Information Processing Systems (NeurIPS), December 2018
