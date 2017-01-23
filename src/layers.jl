using PyCall
using StatsBase

import Base: convert, show

abstract Layer

convert{T<:Layer}(t::Type{T}, x::PyObject) = error("convert(::Type{$T}, ::PyObject) not implemented.")
PyObject{T<:Layer}(l::T) = error("PyObject(::$T) not implemented.")

function StatsBase.weights{T<:Layer}(l::T)
    obj = PyObject(l)
    return obj[:get_weights]()
end

function weights!{T<:Layer}(l::T, W::Array)
    obj = PyObject(l)
    obj[:set_weights](W)
end

function config{T<:Layer}(l::T)
    obj = PyObject(l)
    return obj[:get_config]()
end

function input{T<:Layer}(l::T)
    obj = PyObject(l)
    return Tensor(obj[:input])
end

function input{T<:Layer}(l::T, i::Int)
    obj = PyObject(l)
    return Tensor(obj[:get_input_at](i))
end

function output{T<:Layer}(l::T)
    obj = PyObject(l)
    return Tensor(obj[:output])
end

function output{T<:Layer}(l::T, i::Int)
    obj = PyObject(l)
    return Tensor(obj[:get_output_at](i))
end

function input_shape{T<:Layer}(l::T)
    obj = PyObject(l)
    return obj[:input_shape]
end

function input_shape{T<:Layer}(l::T, i::Int)
    obj = PyObject(l)
    return obj[:get_input_shape_at](i)
end

function output_shape{T<:Layer}(l::T)
    obj = PyObject(l)
    return obj[:output_shape]
end

function output_shape{T<:Layer}(l::T, i::Int)
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
const keras_core_layers = [
    "Dense",
    "Activation",
    "Dropout",
    "Flatten",
    "Reshape",
    "Permute",
    "RepeatVector",
    "Merge",
    "Lambda",
    "ActivityRegularization",
    "Masking",
    "Highway",
    "MaxoutDense",
]

const keras_conv_layers = [
    "Convolution1D",
    "AtrousConvolution1D",
    "Convolution2D",
    "AtrousConvolution2D",
    "SeparableConvolution2D",
    "Deconvolution2D",
    "Convolution3D",
    "Cropping1D",
    "Cropping2D",
    "Cropping3D",
    "UpSampling1D",
    "UpSampling2D",
    "UpSampling3D",
    "ZeroPadding1D",
    "ZeroPadding2D",
    "ZeroPadding3D",
]

const keras_pooling_layers = [
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
]

const keras_local_layers = ["LocallyConnected1D", "LocallyConnected2D"]

const keras_recurrent_layers = [
    "Recurrent",
    "SimpleRNN",
    "GRU",
    "LSTM",
]

const keras_embedding_layers = [ "Embedding" ]

const keras_activation_layers = [
    "LeakyReLU",
    "PReLU",
    "ELU",
    "ParametricSoftplus",
    "ThresholdedReLU",
    "SReLU",
]

const keras_normalization_layers = [ "BatchNormalization" ]
const keras_noise_layers = [ "GaussianNoise", "GaussianDropout" ]
const keras_wrapper_layers = [ "TimeDistributed", "Bidirectional"]

const keras_all_layers = vcat(
    keras_core_layers,
    keras_conv_layers,
    keras_pooling_layers,
    keras_local_layers,
    keras_recurrent_layers,
    keras_embedding_layers,
    keras_activation_layers,
    keras_normalization_layers,
    keras_noise_layers,
    # keras_wrapper_layers, # Not sure how to address wrapping the other layers.
)

for l in keras_all_layers
    layer_name = Symbol(l)

    @eval begin
        type $layer_name <: Keras.Layer
            obj::PyObject

            @doc PyDoc(Keras._layers, Symbol($l)) function $layer_name(args...; kwargs...)
                new(Keras._layers[Symbol($l)](args...; kwargs...))
            end
        end

        #convert(::Type{$(layer_name)}, obj::PyObject) = $layer_name(obj)
        PyObject(layer::$(layer_name)) = layer.obj
        # Base.Docs.doc(layer::$(layer_name)) = Base.Docs.doc(layer.obj)
        function (_::$(layer_name))(args...; kwargs...)
            obj = PyObject(_)
            return obj[:__call__](args...; kwargs...)
        end
    end
end

end
