# Week 01

2015-09-12

## Installation Instructions

We're going to use `tensorflow` and `keras` to run demonstrations.
`tensorflow` is Google's computer library for deep learning (neural networks) and `keras` is a front-end that makes it as intuitive as possible to set up models.
`keras` runs `tensorflow` behind the scenes.
Although `keras` has its own software library, the easiest (and best-documented) way to use it is through the `python` interface.
We will use version 3 of `python` but be aware that some tutorials you find online may use `python` 2, which is subtly different.
If you're comfortable with `python` you may have a better way to set up this installation.

The easiest way to install `keras` and `tensorflow` is to use `conda`; we recommend `miniconda` which you can download [here](https://conda.io/miniconda.html) or using your installer of choice (`apt-get`, `brew`, etc.)
This is package manage for `python` which lets you install other packages and their dependencies.

Once you have `miniconda` working, you need to install a few packages:

- `tensorflow` to give you the back end for computation
- `keras` to build models
- `matplotlib` to make simple plots
- `numpy` to work with arrays of numbers
- `jupyter` to create and use notebooks
- `pandas` for working with tabular and ordered data

Run `conda upgrade --all` to make sure you're up to date.
Run `conda install tensorflow numpy matplotlib jupyter pandas & conda install -c conda-forge keras` to install these packages.

## Tutorials to Get Started

This will be easier for you if you play (briefly is fine!) with `python` and `numpy` a bit.
[This Lesson](http://swcarpentry.github.io/python-novice-inflammation/) by Software Carpentry is a good introduction to working with `python`.
To learn about `jupyter` notebooks, you can look at [this guide to getting started](https://jupyter.readthedocs.io/en/latest/running.html).
If you've used `Rmarkdown`, it's trying to achieve (very broadly) the same thing.
To test that you have everything working, run `jupyter notebook` in the terminal (if anyone knows what to do on Windows please let us know and we will update!).
If you can't get it working, ask someone for help!

## How Far Should I Go?

If you can get a jupyter notebook to open, and have some *basic* ability to read `python`, you'll be good for next Tuesday!
