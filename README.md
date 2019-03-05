# Keras-FlappyBird

A single 200 lines of python code to demostrate DQN with Keras

Please read the following [blog](https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html) for details.

## Showing

![](assets/animation.gif)

## Requestment

-   Python 3.x
-   Tensorflow >= 1.11.0
-   pygame
-   scikit-image

## Install

```bash
$ git clone https://github.com/yanpanlau/Keras-FlappyBird.git
$ cd Keras-FlappyBird
$ python qlearn.py -m run
```

> If you want to train the network from beginning, delete the model.h5 and run with `python qlearn.py -m train`

## TODO

-   [x] Py2 -> Py3
-   [ ] Store model over time