function Metric(f::Function, name=nothing)
    fn = name == nothing ? typeof(f).name.mt.name : name
    fn_wrapper = @eval begin
        f -> begin
            function $(fn)(actual::PyObject, pred::PyObject)
                f(Tensor(actual), Tensor(pred)).o
            end

            return $(fn)
        end
    end

    return PyObject(fn_wrapper(f))
end

export Metric
