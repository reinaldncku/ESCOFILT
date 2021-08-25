# Unsupervised Extractive Summarization-Based Representations for Accurate and Explainable Collaborative Filtering

AKA ESCOFILT. This is the code implementation of [our paper](https://aclanthology.org/2021.acl-long.232/) accepted at the [ACL 2021](https://2021.aclweb.org). If you are going to use the code and/or refer to the paper, please properly cite us. Thanks!

> Pugoy, R.A.D., & Kao, H.Y. (2021). Unsupervised Extractive Summarization-Based Representations for Accurate and Explainable Collaborative Filtering. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 2981–2990). Association for Computational Linguistics.


## Preparatory Codes

* **params.py** - The values for various parameters/hyperparameters can be modified in this file.
* **paths_args.py** - The paths (e.g. path of the dataset) can also be adjusted here.
* **general_codes.py** - Has the function bodies for the training and evaluation phases.
* **recsys_lib20.py** - Contains the functions for obtaining the extractive summary embeddings, the actual collaborative filtering, and other utilities.

## Main Codes

* **[0] Prepare ExSumm Emb.ipynb** - Run this notebook to obtain and save extractive summary embeddings.
* **[1] Run AceCF (Train-Pred).ipynb** - Run this notebook to train and evaluate AceCF.


## Citation
[to follow]
