# Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System
This is our Pytorch implementation for the paper:
> Ding Zou, Wei Wei, Xian-Ling Mao, Ziyang Wang, Minghui Qiu, Feida Zhu, Xin Cao (2022). Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System, [Paper in arXiv](https://arxiv.org/pdf/2204.08807.pdf). In SIGIR'22.


## Introduction
Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System (MCCLK) is a knowledge-aware recommendation solution based on GNN and Contrastive Learning,
proposing a multi-level cross-view contrastive framework to enhance representation learning from multi-faced aspects.

## Requirement
The code has been tested running under Python 3.7.9. The required packages are as follows:
- pytorch == 1.5.0
- numpy == 1.15.4
- scipy == 1.1.0
- sklearn == 0.20.0
- torch_scatter == 2.0.5
- torch_sparse == 0.6.10
- networkx == 2.5

## Usage
The hyper-parameter search range and optimal settings have been clearly stated in the codes (see the parser function in utils/parser.py).
* Train and Test

```
python main.py 
```

## Citation
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{mcclk2022,
  author    = {Zou, Ding and
               Mao, Xian-Ling and
	       Wang, Ziyang and
	       Qiu, Minghui and
	       Zhu, Feida and
	       Cao, Xin},
  title     = {Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System},
  booktitle = {Proceedings of the 45th International {ACM} {SIGIR} Conference on
               Research and Development in Information Retrieval, {SIGIR} 2022, Madrid,
               Spain, July 11-15, 2022.},
  year      = {2022},
}
```



## Dataset

We provide three processed datasets: Book-Crossing, MovieLens-1M, and Last.FM.

We follow the paper " [Ripplenet: Propagating user preferences on the knowledge
graph for recommender systems](https://github.com/hwwang55/RippleNet)." to process data.


|                       |               | Book-Crossing | MovieLens-1M | Last.FM |
| :-------------------: | :------------ | ----------:   | --------: | ---------: |
| User-Item Interaction | #Users        |      17,860   |    6,036  |      1,872 |
|                       | #Items        |      14,967   |    2,445  |      3,846 |
|                       | #Interactions |     139,746   |  753,772  |      42,346|
|    Knowledge Graph    | #Entities     |      77,903   |    182,011|      9,366 |
|                       | #Relations    |          25   |         12|         60 |
|                       | #Triplets     |   151,500     |  1,241,996|     15,518 |


## Reference 
- We partially use the codes of [KGIN](https://github.com/huangtinglin/Knowledge_Graph_based_Intent_Network).
- You could find all other baselines in Github.
