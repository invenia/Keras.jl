# Keras.jl

[![Build Status](https://travis-ci.org/invenia/Keras.jl.svg?branch=master)](https://travis-ci.org/invenia/Keras.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/2s8yo4bkojubnb85/branch/master?svg=true)](https://ci.appveyor.com/project/rofinn/keras-jl/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/invenia/Keras.jl/badge.svg?branch=master)](https://coveralls.io/github/invenia/Keras.jl?branch=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](http://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)


Keras.jl uses [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) to build a julia wrapper around the python neural network library [keras](https://keras.io/).

## Installation

Keras.jl is not a registered package yet so you should install it via:
```julia
julia> Pkg.clone("https://github.com/invenia/Keras.jl")
```

Keras.jl can handle installing [tensorflow](https://www.tensorflow.org/) (v0.9.0) and [keras](https://keras.io/) (v1.0.7) using [Conda.jl](https://github.com/JuliaPy/Conda.jl) and the [conda-forge](https://www.continuum.io/blog/developer-blog/community-conda-forge) channel.

Start by setting `PYTHON` environment variable so that PyCall.jl uses the default python and package for Conda.jl.
```julia
julia> ENV["PYTHON"] = ""
```

Now rebuild PyCall.jl and Keras.jl
```julia
julia> Pkg.build("PyCall")
...

julia> Pkg.build("Keras")
...
```

When building Keras.jl a default keras.json config will be written to `~/.keras/keras.json` for you.


## Getting Started
```julia
julia> using StatsBase

julia> using Keras
Using Tensorflow backend.

julia> import Keras.Layers: Dense, Activation

julia> model = Sequential()
Keras.Sequential(PyObject <keras.models.Sequential object at 0x3186b0590>,Keras.Layer[])

julia> add!(model, Dense(80, input_dim=735))

julia> add!(model, Activation(:relu))

julia> add!(model, Dense(10))

julia> add!(model, Activation(:softmax))

julia> compile!(model; loss=:categorical_crossentropy, optimizer=:sgd, metrics=[:accuracy])

julia> h = fit!(model, rand(1000, 735), rand(1000, 10); nb_epoch=100, batch_size=32, verbose=1)
Epoch 1/100
1000/1000 [==============================] - 0s - loss: 11.5345 - acc: 0.1110
Epoch 2/100
1000/1000 [==============================] - 0s - loss: 11.5048 - acc: 0.1160
Epoch 3/100
1000/1000 [==============================] - 0s - loss: 11.4877 - acc: 0.1150
Epoch 4/100
1000/1000 [==============================] - 0s - loss: 11.4755 - acc: 0.1280
Epoch 5/100
1000/1000 [==============================] - 0s - loss: 11.4691 - acc: 0.1180
Epoch 6/100
1000/1000 [==============================] - 0s - loss: 11.4648 - acc: 0.1270
Epoch 7/100
1000/1000 [==============================] - 0s - loss: 11.4624 - acc: 0.1320
Epoch 8/100
1000/1000 [==============================] - 0s - loss: 11.4605 - acc: 0.1230
Epoch 9/100
1000/1000 [==============================] - 0s - loss: 11.4585 - acc: 0.1200
Epoch 10/100
1000/1000 [==============================] - 0s - loss: 11.4571 - acc: 0.1210
Epoch 11/100
1000/1000 [==============================] - 0s - loss: 11.4557 - acc: 0.1220
Epoch 12/100
1000/1000 [==============================] - 0s - loss: 11.4539 - acc: 0.1280
Epoch 13/100
1000/1000 [==============================] - 0s - loss: 11.4528 - acc: 0.1350
Epoch 14/100
1000/1000 [==============================] - 0s - loss: 11.4512 - acc: 0.1220
Epoch 15/100
1000/1000 [==============================] - 0s - loss: 11.4500 - acc: 0.1330
Epoch 16/100
1000/1000 [==============================] - 0s - loss: 11.4484 - acc: 0.1370
Epoch 17/100
1000/1000 [==============================] - 0s - loss: 11.4472 - acc: 0.1340
Epoch 18/100
1000/1000 [==============================] - 0s - loss: 11.4456 - acc: 0.1350
Epoch 19/100
1000/1000 [==============================] - 0s - loss: 11.4446 - acc: 0.1330
Epoch 20/100
1000/1000 [==============================] - 0s - loss: 11.4430 - acc: 0.1400
Epoch 21/100
1000/1000 [==============================] - 0s - loss: 11.4418 - acc: 0.1410
Epoch 22/100
1000/1000 [==============================] - 0s - loss: 11.4404 - acc: 0.1410
Epoch 23/100
1000/1000 [==============================] - 0s - loss: 11.4389 - acc: 0.1430
Epoch 24/100
1000/1000 [==============================] - 0s - loss: 11.4380 - acc: 0.1460
Epoch 25/100
...
Epoch 100/100
1000/1000 [==============================] - 0s - loss: 11.3092 - acc: 0.2580
PyObject <keras.callbacks.History object at 0x322cd7f10>

julia> evaluate(model, rand(10, 735), rand(10, 10); batch_size=5, verbose=1)
 5/10 [==============>...............] - ETA: 0s2-element Array{Any,1}:
 12.4966
  0.2

julia> predict(model, rand(10, 735); batch_size=5, verbose=1)
5/10 [==============>...............] - ETA: 0s10×10 Array{Float32,2}:
0.100348   0.0992692  0.0923158  0.0851037  0.103129   0.106778   0.11058    0.101286   0.0894175  0.111771
0.0976726  0.0875343  0.108842   0.0852642  0.0820421  0.106634   0.105027   0.101865   0.121497   0.103621
0.0598381  0.10897    0.0837665  0.0733929  0.0725107  0.122981   0.162031   0.144932   0.0795453  0.0920316
0.131148   0.124512   0.087671   0.116357   0.087715   0.085439   0.0887628  0.0977076  0.102328   0.0783594
0.135756   0.131731   0.135302   0.0765197  0.0982455  0.0807826  0.0835469  0.0812529  0.113327   0.0635366
0.0944178  0.072396   0.142516   0.0971909  0.107448   0.0802704  0.0959812  0.113621   0.0858358  0.110323
0.101508   0.102114   0.0758901  0.128942   0.114254   0.110218   0.0886269  0.0907067  0.113456   0.0742862
0.113782   0.0607797  0.101801   0.0774414  0.0919279  0.0721519  0.127951   0.0911597  0.106482   0.156523
0.0964221  0.0636331  0.0747976  0.0916739  0.0871047  0.117788   0.101153   0.133917   0.106753   0.126758
0.131108   0.074069   0.0929551  0.0934865  0.0924049  0.0769063  0.111134   0.115697   0.12413    0.0881087
```

## Running Tests
```julia
julia> Pkg.test("Keras")
```

## API Overview

### Tensors

The `Tensor` type wraps python tensors used by keras (Tensorflow or Theano).
The wrapped PyCall code for working with these `Tensor`s is provided.

Notes:

- Many of the element-wise mathematical operations described [here](https://www.tensorflow.org/api_docs/python/framework/core_graph_data_structures#Tensor.__add__) use the `.` syntax used by julia, so python tensor code like `x + y` would be written as `x .+ y` in julia.
- All methods using the `Tensor` type will perform the appropriate python operation and return the resulting tensor in a new `Tensor` wrapper.

### Models

An abstract `Model` type is defined to describe a minimal `Model` interface.
The `Sequential` model is provided with most of the standard operations provided in the examples; however, more functionality will be added as needed.

Notes:

- Methods like `model.layers()` and `model.get_inputs()` have been replaced with `layers(model)` and `inputs(model)`.
- Mutating methods like `model.fit` and `model.add` use the appropriate julia convention of `fit!(model, ...)` and `add!(model, layer)`.

### Layers, Optimizers & Regularizers

All of the layers, optimizers and regularizers provided within the base keras library have minimal wrappers for convenient object creation and dispatch.
Currently, all args and kwargs for these constructors and functions are passed directly to the python code; however, the original python docstrings are accessible through the help system in juila.

Example)
````julia
help?> Keras.Layers.Dense
Just your regular fully connected NN layer.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_dim=16))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # this is equivalent to the above:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.

````

Notes:

- Methods like `layer.get_weights()` and `layer.set_weights(W)` have been replaced with more julian forms of `weights(layer)` and `weights!(layer, W)`.
- Rather than having separate `layer.get_input()` and `layer.get_input_at(i)` methods we simply dispatch with `input(layer)` and `input(layer, i)` appropriately.

## TODO

1. Wrap saving and loading model state.
1. Migrate to Keras 2.0
1. 100% Test coverage
1. Properly convert python docstrings to julia (low priority)
