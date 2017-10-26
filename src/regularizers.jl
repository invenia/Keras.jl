module Regularizers

import PyCall: PyObject, pycall, PyAny

import ..Keras
import ..Keras: PyDoc

const keras_regularizer_classes = [ "Regularizer", "L1L2"]
const keras_regularizer_aliases = [
    "l1",
    "l2",
    "l1_l2",
]

for r in keras_regularizer_classes
    reg_name = Symbol(r)

    @eval begin
        type $reg_name
            obj::PyObject

            @doc PyDoc(Keras._regularizers, Symbol($r)) function $reg_name(args...; kwargs...)
                new(Keras._regularizers[Symbol($r)](args...; kwargs...))
            end
        end

        # convert(::Type{$(reg_name)}, obj::PyObject) = $reg_name(obj)
        PyObject(reg::$(reg_name)) = reg.obj
        pycall(reg::$(reg_name), args...; kws...) = pycall(reg.obj, args...; kws...)
    end
end

for r in keras_regularizer_aliases
    rf = Symbol(r)

    @eval begin
        @doc PyDoc(Keras._regularizers, Symbol($r)) function $rf(args...; kwargs...)
            return pycall(Keras._regularizers[Symbol($r)], PyAny, args...; kwargs...)
        end
    end
end

end
