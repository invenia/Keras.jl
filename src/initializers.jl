module Initializations

import PyCall: PyObject, pycall

import ..Keras
import ..Keras: PyDoc

const keras_initializer_obj = [
    "Zeros",
    "Ones",
    "Constant",
    "RandomNormal",
    "RandomUniform",
    "TruncatedNormal",
    "VarianceScaling",
    "Orthogonal",
    "Identity",

]

for i in keras_initializer_obj
    init_name = Symbol(i)

    @eval begin
        type $init_name
            obj::PyObject

            @doc PyDoc(Keras._initializers, Symbol($i)) function $init_name(args...; kwargs...)
                new(Keras._initializers[Symbol($i)](args...; kwargs...))
            end
        end

        PyObject(initializer::$(init_name)) = initializer.obj
    end
end

const keras_initializer_funcs = [
    "lecun_uniform",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
]

for i in keras_initializer_funcs
    init_name = Symbol(i)

    @eval begin
        @doc PyDoc(Keras._initializers, Symbol($i)) function $init_name(args...; kwargs...)
            Keras._initializers[Symbol($i)](args...; kwargs...)
        end
    end
end

end
