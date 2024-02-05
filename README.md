# KGRecTest
Official implement of paper "Does Knowledge Graph Really Matter for Recommender Systems?", WWW 2024.

## Requirements
Tested on python 3.9 and Ubuntu 20.04.
1. [pytorch](https://pytorch.org/)==2.0.0
2. [recbole](https://recbole.io/)==1.1.1
3. [lightgbm](https://github.com/microsoft/LightGBM/tree/master/python-package)
4. [xgboost](https://github.com/dmlc/xgboost)
5. [ray](https://www.ray.io/)
6. [thop](https://github.com/Lyken17/pytorch-OpCounter)
7. [torch_scatter](https://github.com/rusty1s/pytorch_scatter/tree/master)
8. [hyperopt](https://github.com/hyperopt/hyperopt)
9. [dgl](https://github.com/dmlc/dgl/)==0.9.1

## Dataset process
Please refer to [recbole](https://recbole.io/) to download the datasets and our paper for details.

## Tips
1. Our results are saved in ```./result```. If you want to run experiments, please move ```./result``` to other place or just delete it.
2. Replace ```np.float``` with ```float``` in ```recbole.evaluator.metrics``` as there is a conflict between recbole and high version of numpy.
3. Run ```python main.py -h``` to see the usage.

## No knowledge experiment example
```shell
python main.py --experiment noknowledge --dataset Movielens-1m --worker_num 4 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_type_list fact --rate 1 --save_dataset --save_dataloaders

python main.py --experiment noknowledge --dataset lastfm --worker_num 4 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_type_list fact --rate 1 --save_dataset --save_dataloaders

python main.py --experiment noknowledge --dataset book-crossing --worker_num 4 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_type_list fact --rate 1 --save_dataset --save_dataloaders

python main.py --experiment noknowledge --dataset Amazon_Books --worker_num 4 --eval_times 5 --model KGAT KGIN KGCN RippleNet CFKG CKE KGNNLS KTUP --test_type_list fact --rate 1 --save_dataset --save_dataloaders --offset 1
```

## False experiment example
```shell
python main.py --experiment false --dataset lastfm --worker_num 4 --rate 0 0.25 0.5 0.75 1.0 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_type_list kg --save_dataset --save_dataloaders

python main.py --experiment false --dataset book-crossing --worker_num 4 --rate 0 0.25 0.5 0.75 1.0 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_type_list kg --save_dataset --save_dataloaders

python main.py --experiment false --dataset Movielens-1m --worker_num 4 --rate 0 0.25 0.5 0.75 1.0 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_type_list kg --save_dataset --save_dataloaders

python main.py --experiment false --dataset Amazon_Books --worker_num 4 --rate 0 0.25 0.5 0.75 1.0 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_type_list kg --save_dataset --save_dataloaders
```

## Decrease experiment example
```shell
python main.py --experiment decrease --dataset Movielens-1m --worker_num 4 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN  --test_type_list fact entity relation --rate 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset lastfm --worker_num 4 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN  --test_type_list fact entity relation --rate 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset Amazon_Books --worker_num 4 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN  --test_type_list fact entity relation --rate 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset book-crossing --worker_num 4 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN  --test_type_list fact entity relation --rate 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset Movielens-1m --worker_num 4 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN  --test_type_list fact --rate 0.0 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset lastfm --worker_num 4 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN  --test_type_list fact --rate 0.0 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset Amazon_Books --worker_num 4 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN  --test_type_list fact --rate 0.0 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset book-crossing --worker_num 4 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN  --test_type_list fact --rate 0.0 --save_dataset --save_dataloaders
```


## Cold-start experiment example
```shell
python main.py --experiment coldstart --dataset book-crossing --worker_num 2 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 5 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 2

python main.py --experiment coldstart --dataset lastfm --worker_num 2 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 5 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 2

python main.py --experiment coldstart --dataset Amazon_Books --worker_num 2 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 5 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 2

python main.py --experiment coldstart --dataset Movielens-1m --worker_num 2 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 5 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 2

python main.py --experiment coldstart --dataset Movielens-1m --worker_num 4 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 3 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 2

python main.py --experiment coldstart --dataset book-crossing --worker_num 2 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 3 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 4

python main.py --experiment coldstart --dataset lastfm --worker_num 2 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 3 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 4

python main.py --experiment coldstart --dataset Amazon_Books --worker_num 2 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 3 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 4


python main.py --experiment coldstart --dataset book-crossing --worker_num 2 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 1 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 4

python main.py --experiment coldstart --dataset lastfm --worker_num 2 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 1 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 4

python main.py --experiment coldstart --dataset Amazon_Books --worker_num 2 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 1 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 4

python main.py --experiment coldstart --dataset Movielens-1m --worker_num 2 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 1 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 4
```

local_server:
```shell
python main.py --experiment coldstart --dataset book-crossing --worker_num 4 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 3 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 2

python main.py --experiment coldstart --dataset lastfm --worker_num 4 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 3 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 2

python main.py --experiment coldstart --dataset Amazon_Books --worker_num 3 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 3 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 2

python main.py --experiment coldstart --dataset Movielens-1m --worker_num 2 --eval_times 5 --model KGAT KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_user_ratio 0.1 --cs_threshold 3 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 2

python main.py --experiment false --dataset lastfm --worker_num 4 --rate 0 0.25 0.5 0.75 1.0 --eval_times 5 --model KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_type_list kg --save_dataset --save_dataloaders

python main.py --experiment false --dataset book-crossing --worker_num 4 --rate 0 0.25 0.5 --eval_times 5 --model KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_type_list kg --save_dataset --save_dataloaders
//








python main.py --experiment coldstart --dataset Amazon_Books --worker_num 3 --eval_times 5 --model KGAT --test_user_ratio 0.1 --cs_threshold 1 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders --offset 2







```


server0:
```shell
python main.py --experiment false --dataset lastfm --worker_num 3 --rate 0 0.25 0.5 0.75 1.0 --eval_times 5 --model KGAT --test_type_list kg --save_dataset --save_dataloaders

python main.py --experiment false --dataset book-crossing --worker_num 3 --rate 0 0.25 0.5 0.75 1.0 --eval_times 5 --model KGAT --test_type_list kg --save_dataset --save_dataloaders

python main.py --experiment false --dataset Movielens-1m --worker_num 3 --rate 0 0.25 0.5 0.75 1.0 --eval_times 5 --model KGAT --test_type_list kg --save_dataset --save_dataloaders

python main.py --experiment false --dataset Amazon_Books --worker_num 2 --rate 0 0.25 0.5 0.75 1.0 --eval_times 5 --model KGAT --test_type_list kg --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset Amazon_Books --worker_num 2 --eval_times 5 --model KGAT --test_type_list entity relation --rate 0.25 0.5 --save_dataset --save_dataloaders
//
python main.py --experiment coldstart --dataset Movielens-1m --worker_num 2 --eval_times 5 --model KGAT --test_user_ratio 0.1 --cs_threshold 5 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders 


```

server1:
```shell
python main.py --experiment decrease --dataset Movielens-1m --worker_num 3 --eval_times 5 --model KGAT --test_type_list fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset lastfm --worker_num 3 --eval_times 5 --model KGAT --test_type_list fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset Amazon_Books --worker_num 2 --eval_times 5 --model KGAT --test_type_list fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset book-crossing --worker_num 3 --eval_times 5 --model KGAT --test_type_list fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset Amazon_Books --worker_num 2 --eval_times 5 --model KGAT --test_type_list entity relation --rate 0.75 1.0 --save_dataset --save_dataloaders

python main.py --experiment coldstart --dataset Amazon_Books --worker_num 2 --eval_times 5 --model KGAT --test_user_ratio 0.1 --cs_threshold 5 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

//
python main.py --experiment false --dataset book-crossing --worker_num 4 --rate 0.5 --eval_times 5 --model CFKG CKE KGNNLS KTUP KGIN --test_type_list kg --save_dataset --save_dataloaders

python main.py --experiment coldstart --dataset Movielens-1m --worker_num 2 --eval_times 5 --model KGAT --test_user_ratio 0.1 --cs_threshold 1 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders


```

server2(1gpu):
```shell
python main.py --experiment noknowledge --dataset Movielens-1m --worker_num 1 --eval_times 5 --model KGAT --test_type_list fact --rate 1 --save_dataset --save_dataloaders

python main.py --experiment noknowledge --dataset lastfm ---worker_num 1 --eval_times 5 --model KGAT --test_type_list fact --rate 1 --save_dataset --save_dataloaders

python main.py --experiment noknowledge --dataset book-crossing --worker_num 1 --eval_times 5 --model KGAT --test_type_list fact --rate 1 --save_dataset --save_dataloaders

python main.py --experiment noknowledge --dataset Amazon_Books --worker_num 1 --eval_times 5 --model KGAT --test_type_list fact --rate 1 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset lastfm --worker_num 1 --eval_times 5 --model KGAT --test_type_list entity relation --rate 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset Movielens-1m --worker_num 1 --eval_times 5 --model KGAT --test_type_list entity --rate 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

python main.py --experiment coldstart --dataset book-crossing --worker_num 1 --eval_times 5 --model KGAT --test_user_ratio 0.1 --cs_threshold 5 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

python main.py --experiment coldstart --dataset lastfm --worker_num 1 --eval_times 5 --model KGAT --test_user_ratio 0.1 --cs_threshold 5 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

//
python main.py --experiment false --dataset book-crossing --worker_num 1 --rate 0.75 --eval_times 5 --model KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_type_list kg --save_dataset --save_dataloaders

python main.py --experiment false --dataset book-crossing --worker_num 1 --rate 0.5 --eval_times 5 --model RippleNet --test_type_list kg --save_dataset --save_dataloaders



```
server3(1gpu):
```shell
python main.py --experiment decrease --dataset Movielens-1m --worker_num 1 --eval_times 5 --model RippleNet CFKG --test_type_list entity --rate 1 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset Movielens-1m --worker_num 1 --eval_times 5 --model KGIN --test_type_list fact --rate 1 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset lastfm --worker_num 1 --eval_times 5 --model  KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN  --test_type_list fact --rate 0.0 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset book-crossing --worker_num 1 --eval_times 5 --model KGAT --test_type_list entity relation --rate 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

python main.py --experiment decrease --dataset Movielens-1m --worker_num 1 --eval_times 5 --model KGAT --test_type_list relation --rate 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

python main.py --experiment coldstart --dataset book-crossing --worker_num 1 --eval_times 5 --model KGAT --test_user_ratio 0.1 --cs_threshold 1 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

python main.py --experiment coldstart --dataset lastfm --worker_num 1 --eval_times 5 --model KGAT --test_user_ratio 0.1 --cs_threshold 1 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

//
python main.py --experiment false --dataset book-crossing --worker_num 1 --rate 1.0 --eval_times 5 --model KGCN RippleNet CFKG CKE KGNNLS KTUP KGIN --test_type_list kg --save_dataset --save_dataloaders

```

server4(gpu1)
```shell
python main.py --experiment coldstart --dataset Movielens-1m --worker_num 1 --eval_times 5 --model KGAT --test_user_ratio 0.1 --cs_threshold 5 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders 

```

server5(gpu1)
```shell
python main.py --experiment coldstart --dataset Movielens-1m --worker_num 1 --eval_times 5 --model KGAT --test_user_ratio 0.1 --cs_threshold 1 --test_type_list random d_fact --rate 0 0.25 0.5 0.75 1 --save_dataset --save_dataloaders

```