module Callbacks

import PyCall: PyObject, pycall

import ..Keras
import ..Keras: PyDoc

const keras_callbacks = [
    "BaseLogger",
    "ProgbarLogger",
    "History",
    "ModelCheckpoint",
    "EarlyStopping",
    "RemoteMonitor",
    "LearningRateScheduler",
    "TensorBoard",
    "ReduceLROnPlateau",
    "CSVLogger",
    "LambdaCallback",
]

for c in keras_callbacks
    cb_name = Symbol(c)

    @eval begin
        struct $cb_name
            obj::PyObject

            @doc PyDoc(Keras._callbacks, Symbol($c)) function $cb_name(args...; kwargs...)
                new(Keras._callbacks[Symbol($c)](args...; kwargs...))
            end
        end

        PyObject(callback::$(cb_name)) = callback.obj
    end
end

end
