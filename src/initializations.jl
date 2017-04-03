module Initializations

import PyCall: PyObject, pycall

import ..Keras
import ..Keras: PyDoc

const keras_initializer_funcs = [
    "uniform",
    "lecun_uniform",
    "normal",
    "identity",
    "orthogonal",
    "zero",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
]

for i in keras_initializer_funcs
    init_name = Symbol(i)

    @eval begin
        @doc PyDoc(Keras._initializations, Symbol($i)) function $init_name(args...; kwargs...)
            Keras._initializations[Symbol($i)](args...; kwargs...)
        end
    end
end

end
