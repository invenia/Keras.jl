module Regularizers

import PyCall: PyObject, pycall

import ..Keras
import ..Keras: PyDoc

const keras_regularizers = [ "WeightRegularizer", "ActivityRegularizer" ]
const keras_regularizer_shortcuts = [
    "l1",
    "l2",
    "l1l2",
    "activity_l1",
    "activity_l2",
    "activity_l1l2",
]

for r in keras_regularizers
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

for r in keras_regularizer_shortcuts
    rf = Symbol(r)

    @eval begin
        @doc PyDoc(Keras._regularizers, Symbol($r)) function $rf(args...; kwargs...)
            return pycall(Keras._regularizers[Symbol($r)], PyAny, args...; kwargs...)
        end
    end
end

end
