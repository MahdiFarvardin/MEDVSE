# MEDVSE - Efficient Deep Learning-based Estimation of the Vital Signs on Smartphones

Official repository of "[Efficient Deep Learning-based Estimation of the Vital Signs on Smartphones](https://arxiv.org/abs/2204.08989)".

This repository contains the code and the proposed dataset (MTHS)

## Checklist
- [x] MTHS dataset
- [x] Finger Videos (New!)
- [x] Code
- [ ] Webpage 

## Code

* Codes are included in the `code` folder. Please refer to its `Readme.md` for more detailed information. 
## Dataset - MTHS: 
* This folder contains our dataset
* Each subject has two `.npy` files: mean RGB signals as `signal_x.npy` and ground truth labels as `label_x.npy`, where `x` is the patient id.
* `signal_x.npy` contains the mean signals ordered as R, G, and then B sampled at 30Hz.
* `label_x.npy` contains the ground truth data ordered as HR(bpm) - SpO2(%) Sampled at 1Hz.

### New! - Fingertip videos
Due to some requests we now provide the raw fingertip videos. 

For downloading videos, please send us an email with your academic email containing your Gmail address. 

## Donation :)

If you find our dataset useful please consider donation, it would help us a lot. 

btc:
- bc1qtgtflqv0laapltmwczfg8ree70mv90fcwvvsd4

eth:
- 0xCa432902f1270AD076814cD77E03Aef2D09dAc19

usdt (trc20):
- TYRPhPT5BZTvn4bcYY4xguL1FSVwchHHrN

## License
This project's code is released under the MIT license.
Note that the dataset is released under the [CC BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/4.0/). 



## Citation
If you use our dataset or find this repository helpful, please consider citing:

```
@article{samavati2022efficient,
  title={Efficient deep learning-based estimation of the vital signs on smartphones},
  author={Samavati, Taha and Farvardin, Mahdi and Ghaffari, Aboozar},
  journal={arXiv preprint arXiv:2204.08989},
  year={2022}
}
```

## Contact emails
* tahasamavati12@gmail.com
* mahdi.farvardin@gmail.com
