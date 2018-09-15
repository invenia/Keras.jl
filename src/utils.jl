"""
    metric(f::Function; name=nothing) -> Function

Converts a julia function `f` to a python metric function to be used by Keras.
Assumes that `f` is of the form `f(::Tensor, ::Tensor) -> Tensor`
"""
@pydef mutable struct metric
    function __init__(self, f::Function, name=nothing)
        self[:f] = f
        self[:__name__] = name == nothing ? typeof(f).name.mt.name : name
    end

    function __call__(self, x::PyObject, y::PyObject)
        self[:f](Tensor(x), Tensor(y)).o
    end
end
