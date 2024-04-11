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

## ChangeLog
<details>
  <summary>Click to see change logs</summary>
  
  #### 2024.04.11 update
  - [x] Multiple train/validation/test datasets support
  - [x] Introducing Register mechanism to further clean codes
  - [ ] kfold
</details>

## Usage
#### 1. Add new dataset
1. Create a new dataset class in `datasets` folder, for example, namely `mydatset.py`
2. Import `Register` class in the beginning of `mydataset.py` and register the dataset class using `@Registry.register_dataset()` decorator
    ```python
    # other import statement ...
    from src import Registry
    # other import statement ...

    # other codes ...

    @Registry.register_dataset()
    class MINISTDataset(data.Dataset):
        ...
    ```
    > You can also register collate_function using `@Registry.register_collate()`, examples can be found in [dataset_example](src/datasets/minist_dataset.py)

#### 2. Add new model
1. Create a new package in `src` folder, for example, namely `my_model`, and create a new model class in `my_model` folder, namely `my_model.py`
2. Import `Register` class in the beginning of `my_model.py` and register the model class using `@Registry.register_module()` decorator
    ```python
    # other import statement ...
    from src import Registry
    # other import statement ...

    # other codes ...

    @Registry.register_module()
    class MyNet(nn.Module):
        ...
    ```
    If you implement your own lightning interface, namley define a `pl.LightningModule` in `my_model/interface.py`. You have to add `__PKG__` in config file to specify the package name to make it work.

    ```yaml
    # for example
    __PKG__: my_model
    model_name: &MODEL_NAME MyNet
    dataset_name: &DATASET_NAME MINISTDataset
    ```
    > If no `__PKG__` is defined, the default interface in `src/interface.py` will be used.


### Training

```shell
python main.py --cfg configs/config.yaml
```

### Validation

```shell
python main.py --cfg configs/config.yaml --mode val --vis --save_pred
```

### Test

```shell
python main.py --cfg configs/config.yaml --mode test --vis --save_pred
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
