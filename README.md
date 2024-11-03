# AncientGlyphNet
![image-20241103164332181](.\README.assets\image-20241103164332181.png)

## Install Using Conda

```
conda create -n myEnvs python=3.9
pip install -r requirement.txt  // maybe use 清华源
conda install pytorch torchvision
git clone ****.git
```

## Data Preparation

you should make a train.txt,follows:

```
./datasets/train/img/**.jpg ./datasets/train/gt/**.txt
```

The groundtruth can be `.txt` files, with the following format:

```
x1, y1, x2, y2, x3, y3, x4, y4, zi
```

## Train

```
python ./tools/train.py
```

## Test

```
python ./tools/predict.py
```





**If this repository helps you，please star it. Thanks.**
