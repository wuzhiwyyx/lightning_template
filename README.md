# MyNet
This responsitory demonstrates the file structure of pytorch-lightning based deep model. It takes hand-writing-digits classification as example.

## Introduction

Classification model is composed of two convolution layers and two fully connected layers.

![pytorch-lightning](assets/images/lightning.png)

## Install

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Usage


### Training

```shell
python scrips/main.py --cfg configs/config.yaml
```

### Validation

```shell
python scripts/main.py --cfg configs/config.yaml --mode val --vis --save_pred
```

### Test

```shell
python scripts/main.py --cfg configs/config.yaml --mode test --vis --save_pred
```

## Citation

(Example) Please cite the following papers [AlexNet](https://dl.acm.org/doi/pdf/10.1145/3065386):

```text
@article{krizhevsky2017imagenet,
  title={Imagenet classification with deep convolutional neural networks},
  author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  journal={Communications of the ACM},
  volume={60},
  number={6},
  pages={84--90},
  year={2017},
  publisher={AcM New York, NY, USA}
}
```

## Reference

* http://yann.lecun.com/exdb/mnist/

## Contributing

PRs accepted.

## License

Licensed under the [MIT](LICENSE) License.
