# Week 02

Here's what we have so far:

  - `environment.yml`: you can install a custom conda environment (see resources in slack channel) that has all the packages we're using by running (in the terminal!) `conda env create --file environment.yml`. To activate it (and use those packages) run `source activate cwc_ann`. This has been tested on at least one computer (James) and worked great! Now you can type `source activate cwc_ann` when you're working on our projects here, and `source deactivate` to go back to your other packages when you're done.
- `XOR-numpy-network.ipynb`: from Peter Belhumeur's class (though adapted for python 3) this programs a hidden layer  neural network in python (`numpy`) -- no tensorflow. If you want to understand what this hidden layer is doing, this is worth going through. We won't go through it next Tuesday.
- `00-Sandbox.ipynb` is a space for building your own models. Probably makes sense to do this after working through some of the models below.
- `01-GettingStartedSequentialModel.ipynb` is a very simple notebook where we learn about the `Sequential()` model in `keras`
- `02-MNIST-MLP.ipynb` uses a sequential model to build a multi-layer perceptron to classify handwritten digits
- `03-MNIST-CNN.ipynb` uses the same MNIST handwritten digit data but uses a convolutional model -- which preserves spatial patterns in the data -- for classification.
