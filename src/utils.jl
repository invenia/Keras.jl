function metric(f::Function)
    function inner(actual::PyObject, pred::PyObject)
        return f(Tensor(actual), Tensor(pred)).o
    end

    py_func = PyObject(inner)
    py_func[:__name__] = typeof(f).name.mt.name
    return py_func
end

export metric
