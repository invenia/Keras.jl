using PyCall
using StatsBase

import Base: convert, show

abstract type Layer end

convert(t::Type{T}, x::PyObject) where {T<:Layer} = error("convert(::Type{$T}, ::PyObject) not implemented.")
PyObject(l::T) where {T<:Layer} = error("PyObject(::$T) not implemented.")

function StatsBase.weights(l::T) where {T<:Layer}
    obj = PyObject(l)
    return obj[:get_weights]()
end

function weights!(l::T, W::Array) where {T<:Layer}
    obj = PyObject(l)
    obj[:set_weights](W)
end

function config(l::T) where {T<:Layer}
    obj = PyObject(l)
    return obj[:get_config]()
end

function input(l::T) where {T<:Layer}
    obj = PyObject(l)
    return Tensor(obj[:input])
end

function input(l::T, i::Int) where {T<:Layer}
    obj = PyObject(l)
    return Tensor(obj[:get_input_at](i))
end

function output(l::T) where {T<:Layer}
    obj = PyObject(l)
    return Tensor(obj[:output])
end

function output(l::T, i::Int) where {T<:Layer}
    obj = PyObject(l)
    return Tensor(obj[:get_output_at](i))
end

function input_shape(l::T) where {T<:Layer}
    obj = PyObject(l)
    return obj[:input_shape]
end

function input_shape(l::T, i::Int) where {T<:Layer}
    obj = PyObject(l)
    return obj[:get_input_shape_at](i)
end

function output_shape(l::T) where {T<:Layer}
    obj = PyObject(l)
    return obj[:output_shape]
end

function output_shape(l::T, i::Int) where {T<:Layer}
    obj = PyObject(l)
    return obj[:get_output_shape_at](i)
end

export Layer, weights!, config, input, output, input_shape, output_shape

module Layers

import PyCall: PyObject, pycall

import ..Keras
import ..Keras: PyDoc

#===========================================
Autogenerating Keras layer wrappers.
===========================================#
const keras_layers = Dict{String, Array}(
    "core" => [
        "Dense",
        "Activation",
        "Dropout",
        "Flatten",
        "Reshape",
        "Permute",
        "RepeatVector",
        "Lambda",
        "ActivityRegularization",
        "Masking",
    ],
    "convolutional" => [
        "Conv1D",
        "Conv2D",
        "SeparableConv2D",
        "Conv2DTranspose",
        "Conv3D",
        "Cropping1D",
        "Cropping2D",
        "Cropping3D",
        "UpSampling1D",
        "UpSampling2D",
        "UpSampling3D",
        "ZeroPadding1D",
        "ZeroPadding2D",
        "ZeroPadding3D",
    ],
    "pooling" => [
        "MaxPooling1D",
        "MaxPooling2D",
        "MaxPooling3D",
        "AveragePooling1D",
        "AveragePooling2D",
        "AveragePooling3D",
        "GlobalMaxPooling1D",
        "GlobalMaxPooling2D",
        "GlobalAveragePooling1D",
        "GlobalAveragePooling2D",
    ],
    "local" => [
        "LocallyConnected1D",
        "LocallyConnected2D"
    ],
    "recurrent" => [
        "Recurrent",
        "SimpleRNN",
        "GRU",
        "LSTM",
    ],
    "embeddings" => [
        "Embedding",
    ],
    "merge" => [
        "Add",
        "Multiply",
        "Average",
        "Maximum",
        "Concatenate",
        "Dot",
    ],
    "advanced_activations" => [
        "LeakyReLU",
        "PReLU",
        "ELU",
        "ThresholdedReLU",
    ],
    "normalization" => [
        "BatchNormalization",
    ],
    "noise" => [
        "GaussianNoise",
        "GaussianDropout",
    ],
    "wrappers" => [
        "TimeDistributed",
        "Bidirectional",
    ],
)

for (submod, layers) in keras_layers
    for l in layers
        layer_name = Symbol(l)

        @eval begin
            struct $layer_name <: Keras.Layer
                obj::PyObject

                @doc PyDoc(Keras._layers, Symbol($l)) function $layer_name(args...; kwargs...)
                    new(Keras._layers[Symbol($l)](args...; kwargs...))
                end
            end

            #convert(::Type{$(layer_name)}, obj::PyObject) = $layer_name(obj)
            PyObject(layer::$(layer_name)) = layer.obj
            # Base.Docs.doc(layer::$(layer_name)) = Base.Docs.doc(layer.obj)
            function (layer::$(layer_name))(args...; kwargs...)
                obj = PyObject(layer)
                return obj[:__call__](args...; kwargs...)
            end
        end
    end
end

const keras_merge_funcs = [
    "add",
    "multiply",
    "average",
    "maximum",
    "concatenate",
    "dot",
]

for l in keras_merge_funcs
    layer_name = Symbol(l)

    @eval begin
        @doc PyDoc(Keras._layers, Symbol($l)) function $layer_name(args...; kwargs...)
            Keras._layers[Symbol($l)](args...; kwargs...)
        end
    end
end

end
