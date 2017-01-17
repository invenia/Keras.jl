using Keras
using StatsBase

using Base.Test

import Keras.Layers: Dense, Activation

@testset "Keras" begin
    @testset "Basic Usage" begin
        model = Sequential()
        add!(model, Keras.Layers.Dense(80, input_dim=735))
        add!(model, Keras.Layers.Activation(:relu))
        add!(model, Keras.Layers.Dense(10))
        add!(model, Keras.Layers.Activation(:softmax))
        compile!(
            model;
            loss=:categorical_crossentropy,
            optimizer=:sgd,
            metrics=[:accuracy]
        )

        h = fit!(model, rand(1000, 735), rand(1000, 10); nb_epoch=100, batch_size=32, verbose=0)
    end
end
