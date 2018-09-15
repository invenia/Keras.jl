using StatsBase

abstract type Model end

layers(m::T) where {T<:Model} = error("layers(::$T) not implemented.")

inputs(m::T) where {T<:Model} = error("inputs(::$T) not implemented.")

outputs(m::T) where {T<:Model} = error("outputs(::$T) not implemented.")

Base.getindex(m::T, i::Int) where {T<:Model} = error("getindex(::$T, ::Int) not implemented")

function StatsBase.fit!(m::T, args...; kwargs...) where {T<:Model}
    error("fit(::$T, args...; kwargs...) not implemented.")
end

function evaluate(m::T, args...; kwargs...) where {T<:Model}
    error("evaluate(::$T, args...; kwargs...) not implemented.")
end

function StatsBase.predict(m::T, args...; kwargs...) where {T<:Model}
    error("predict(::$T, args...; kwargs...) not implemented.")
end

#= TODO:
- train_on_batch
- predict_on_batch
- fit_generator
- evaluate_generator
- predict_generator
- get_layer
=#

@doc PyDoc(Keras._models, :Sequential) struct Sequential <: Keras.Model
    obj::PyObject
    layers::Array{Keras.Layer}

    function Sequential()
        new(Keras._models[:Sequential](), Keras.Layer[])
    end

    function Sequential(layers::Array{Keras.Layer})
        new(
            Keras._models[:Sequential]([PyObject(l) for l in layers]),
            layers
        )
    end
end

Keras.layers(m::Sequential) = m.layers

Keras.inputs(m::Sequential) = [Tensor(i) for i in m.obj[:inputs]]

Keras.outputs(m::Sequential) = [Tensor(o) for o in m.obj[:outputs]]

Base.getindex(m::Sequential, i::Int) = m.layers[i]

function add!(m::Sequential, l::Keras.Layer)
    push!(m.layers, l)
    m.obj[:add](PyObject(l))
end

function compile!(m::Sequential, args...; kwargs...)
    m.obj[:compile](args...; kwargs...)
end

function StatsBase.fit!(m::Sequential, args...; verbose=0, kwargs...)
    m.obj[:fit](args...; verbose=verbose, kwargs...)
end

function evaluate(m::Sequential, X, y; batch_size=32, verbose=0)
    return m.obj[:evaluate](X, y, batch_size=batch_size, verbose=verbose)
end

function StatsBase.predict(m::Sequential, X; batch_size=32, verbose=0)
    return m.obj[:predict](X, batch_size=batch_size, verbose=verbose)
end

export Model, Sequential, layers, inputs, outputs, evaluate, add!, compile!
