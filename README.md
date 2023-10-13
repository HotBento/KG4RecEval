# KGRecTest
Official implement of paper "Does Knowledge Graph Really Matter for Recommender Systems?", NIPs 2023.

## Requirements
Tested on python 3.9 and Ubuntu 20.04.
1. [pytorch](https://pytorch.org/)==2.0.0
2. [recbole](https://recbole.io/)==1.1.1
3. [lightgbm](https://github.com/microsoft/LightGBM/tree/master/python-package)
4. [xgboost](https://github.com/dmlc/xgboost)
5. [ray](https://www.ray.io/)
6. [thop](https://github.com/Lyken17/pytorch-OpCounter)
7. [torch_scatter](https://github.com/rusty1s/pytorch_scatter/tree/master)

## Dataset process
You can simply use the processed dataset provided by us. If you want to process the dataset yourself, please refer to [recbole](https://recbole.io/) and our paper for details.

## Tips
1. Our results are saved in ```./result```. If you want to run experiments, please move ```./result``` to other place or just delete it.
2. Replace ```np.float``` with ```float``` in ```recbole.evaluator.metrics``` as there is a conflict between recbole and high version of numpy.
3. Run ```python main.py -h``` to see the usage.

## False experiment example
```shell
python main.py --experiment false --dataset movielens-100k --worker_num 1 --rate 0 0.25 0.5 0.75 1.0 --eval_times 50 --model KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_type_list kg
```

## Decrease experiment example
```shell
python main.py --experiment decrease --dataset Amazon_Books-part --worker_num 1 --eval_times 50 --model KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_type_list fact entity relation --rate 0.25 0.5 0.75 1
```
Note that rates in decrease experiment and KG size experiment are remained rate of entities/facts/relations.

## KG size experiment example
```shell
python main.py --experiment size --dataset lfm1b-tracks-part --worker_num 1 --rate 0.25 0.5 0.75 1 1.25 1.5 --eval_times 50 --model KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_type_list relation entity fact
```

## Cold-start experiment example
```shell
python main.py --experiment coldstart --dataset movielens-100k --worker_num 1 --eval_times 50 --model KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 5 --test_type_list random --rate 0 0.25 0.5 0.75 1
```
## No knowledge experiment example
```shell
python main.py --experiment noknowledge --dataset movielens-100k --worker_num 4 --eval_times 50 --model KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN MCRec --test_type_list fact --rate 1
```

python main.py --experiment coldstart --dataset movielens-100k --worker_num 4 --eval_times 50 --model RippleNet KGIN MCRec --test_user_ratio 0.1 --cs_threshold 1 --test_type_list d_fact d_entity d_relation --rate 0.25 0.5 0.75 1 --offset 0

python main.py --experiment coldstart --dataset movielens-100k --worker_num 4 --eval_times 50 --model RippleNet KGIN MCRec --test_user_ratio 0.1 --cs_threshold 3 --test_type_list d_fact d_entity d_relation --rate 0.25 0.5 0.75 1 --offset 0

python main.py --experiment coldstart --dataset movielens-100k --worker_num 4 --eval_times 50 --model RippleNet KGIN MCRec --test_user_ratio 0.1 --cs_threshold 5 --test_type_list d_fact d_entity d_relation --rate 0.25 0.5 0.75 1 --offset 0

python main.py --experiment coldstart --dataset Amazon_Books-part  --worker_num 4 --eval_times 50 --model RippleNet KGIN MCRec --test_user_ratio 0.1 --cs_threshold 1 --test_type_list d_fact d_entity d_relation --rate 0.25 0.5 0.75 1 --offset 0

python main.py --experiment coldstart --dataset Amazon_Books-part  --worker_num 4 --eval_times 50 --model RippleNet KGIN MCRec --test_user_ratio 0.1 --cs_threshold 3 --test_type_list d_fact d_entity d_relation --rate 0.25 0.5 0.75 1 --offset 0

python main.py --experiment coldstart --dataset Amazon_Books-part  --worker_num 2 --eval_times 50 --model RippleNet KGIN MCRec --test_user_ratio 0.1 --cs_threshold 5 --test_type_list d_fact d_entity d_relation --rate 0.25 0.5 0.75 1 --offset 4

python main.py --experiment coldstart --dataset lfm1b-tracks-part --worker_num 2 --eval_times 50 --model RippleNet KGIN MCRec --test_user_ratio 0.1 --cs_threshold 1 --test_type_list d_fact d_entity d_relation --rate 0.25 0.5 0.75 1 --offset 4

python main.py --experiment coldstart --dataset lfm1b-tracks-part --worker_num 1 --eval_times 50 --model RippleNet KGIN MCRec --test_user_ratio 0.1 --cs_threshold 3 --test_type_list d_fact d_entity d_relation --rate 0.25 0.5 0.75 1 --offset 7

python main.py --experiment coldstart --dataset lfm1b-tracks-part --worker_num 1 --eval_times 50 --model RippleNet KGIN MCRec --test_user_ratio 0.1 --cs_threshold 5 --test_type_list d_fact d_entity d_relation --rate 0.25 0.5 0.75 1 --offset 7

python main.py --experiment noknowledge --dataset movielens-100k --worker_num 4 --eval_times 50 --model KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN MCRec --test_type_list fact --rate 1

python main.py --experiment noknowledge --dataset Amazon_Books-part --worker_num 4 --eval_times 50 --model KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN MCRec --test_type_list fact --rate 1

python main.py --experiment noknowledge --dataset lfm1b-tracks-part --worker_num 4 --eval_times 50 --model KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN MCRec --test_type_list fact --rate 1

python main.py --experiment decrease --dataset Amazon_Books-part --worker_num 4 --eval_times 50 --model RippleNet KGIN MCRec --test_type_list fact entity relation --rate 0.25 0.5 0.75 1

python main.py --experiment decrease --dataset lfm1b-tracks-part --worker_num 4 --eval_times 50 --model RippleNet KGIN MCRec --test_type_list fact entity relation --rate 0.25 0.5 0.75 1

python main.py --experiment size --dataset movielens-100k --worker_num 6 --rate 0.25 0.5 0.75 1 1.25 1.5 --eval_times 50 --model MCRec RippleNet KGIN --test_type_list fact

python main.py --experiment size --dataset Amazon_Books-part  --worker_num 6 --rate 0.25 0.5 0.75 1 1.25 1.5 --eval_times 50 --model MCRec RippleNet KGIN --test_type_list fact

python main.py --experiment size --dataset lfm1b-tracks-part --worker_num 2 --rate 0.25 0.5 0.75 1 1.25 1.5 --eval_times 50 --model MCRec RippleNet KGIN --test_type_list fact

python main.py --experiment coldstart --dataset Amazon_Books-part  --worker_num 8 --eval_times 50 --model KGCN CFKG CKE KGNNLS KTUP --test_user_ratio 0.1 --cs_threshold 1 --test_type_list d_fact d_entity d_relation --rate 0.25 0.5 0.75 1 --offset 0