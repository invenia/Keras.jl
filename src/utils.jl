@pydef type Metric
    __init__(self, f::Function, name=nothing) = (
        self[:f] = f;
        self[:__name__] = name == nothing ? typeof(f).name.mt.name : name
    )
   __call__(self, x::PyObject, y::PyObject) = self[:f](Tensor(x), Tensor(y)).o
end

export Metric
