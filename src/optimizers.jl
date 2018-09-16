module Optimizers

import PyCall: PyObject, pycall

import ..Keras
import ..Keras: PyDoc

const keras_optimizers = [
    "SGD",
    "RMSprop",
    "Adagrad",
    "Adadelta",
    "Adam",
    "Adamax",
    "Nadam",
]

for o in keras_optimizers
    opt_name = Symbol(o)

    @eval begin
        struct $opt_name
            obj::PyObject

            @doc PyDoc(Keras._optimizers, Symbol($o)) function $opt_name(args...; kwargs...)
                new(Keras._optimizers[Symbol($o)](args...; kwargs...))
            end
        end

        # convert(::Type{$(opt_name)}, obj::PyObject) = $opt_name(obj)
        PyObject(opt::$(opt_name)) = opt.obj
    end
end

end
