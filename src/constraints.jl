module Constraints

import PyCall: PyObject, pycall

import ..Keras
import ..Keras: PyDoc

const keras_constraints = ["max_norm", "non_neg", "unit_norm"]

for c in keras_constraints
    const_name = Symbol(c)

    @eval begin
        @doc PyDoc(Keras._constraints, Symbol($c)) function $const_name(args...; kwargs...)
            Keras._constraints[Symbol($c)](args...; kwargs...)
        end
    end
end

end
