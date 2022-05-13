# Codes for MTVital
Official tf-keras implementation of "Efficient Deep Learning-based Estimation of the Vital Signs on Smartphones".

# Usage
For training the models use 'train.py' and for evaluating use 'evaluate.py'. These scripts support CMLs that are described in the following.

## Example training
```terminal 
python train.py --mode='hr' --dataset='mths' --downsample=2 --timelen=10 --batchsize=32 --epochs=125 --testsize=0.2 --valsize=0.15 --savedir='./'
```


## Example evaluating
```terminal 
python evaluate.py --mode='hr' --dataset='mths' --downsample=2 --timelen=10  --testsize=0.2 --valsize=0.15 --savedir='./'
```

Arguments:
*  ```mode```: one of ```hr``` or ```spo2```
*  ```dataset```: The dataset name (default is ```mths```)
*  ```downsample```: The down sample factor for ppg signals
*  ```timelen```: The time length of each input signal to the neural net
*  ```batchsize```: Batch size
*  ```epochs```: Number of training epochs
*  ```testsize```: Testset proportion (of the whole data)
*  ```valsize```: Validation set proportion (of the train data)
*  ```savedir```: Specifies a directory to save the trained models


# Citation
Please cite our work in your project:
```
@misc{2204.08989,
Author = {Taha Samavati and Mahdi Farvardin},
Title = {Efficient Deep Learning-based Estimation of the Vital Signs on Smartphones},
Year = {2022},
Eprint = {arXiv:2204.08989},
}
```
[Efficient Deep Learning-based Estimation of the Vital Signs on Smartphones](https://arxiv.org/abs/2204.08989)
