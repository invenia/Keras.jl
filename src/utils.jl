"""
For now we support 3 different ways of wrapping metric functions in julia.
All 3 methods work in julia v0.5, but fail for various reasons in v0.6.

TODO: Check to see how PyCall.jl fixes this issue in the future.
"""
#=
Defines a callable Metric object which was useful when
PyObject(func) didn't have an `__name__` attribute.
=#
@pydef type metric
    __init__(self, f::Function, name=nothing) = (
        self[:f] = f;
        self[:__name__] = name == nothing ? typeof(f).name.mt.name : name
    )
   __call__(self, x::PyObject, y::PyObject) = self[:f](Tensor(x), Tensor(y)).o
end

# """
# Simply wraps the supplied function, but runs into the world issue on v0.6
# because we're calling a function that was defined in the same scope.
# """
# function metric(f::Function, name=nothing)
#     fn = name == nothing ? typeof(f).name.mt.name : name
#     fn_wrapper = @eval begin
#         f -> begin
#             function $(fn)(actual::PyObject, pred::PyObject)
#                 f(Tensor(actual), Tensor(pred)).o
#             end
#
#             return $(fn)
#         end
#     end
#
#     return PyObject(fn_wrapper(f))
# end

# """
# Rebuilds a function definition with the wrapped code which by passes the
# world issue on v0.6, but still fails with a method error that is currently
# causing tests in PyCall.jl to fail. Hopefully, that issue will be fixed.
#
# NOTE: This macro is nice for specifying a function as a metric once, but
# may make unit testing that function slightly more annoying
#
# Example)
# ```julia
# a = rand(4)
# b = rand(4)
#
# a_tensor = variable(a)
# b_tensor = variable(b)
#
# expected = op(a, b)
# result = Keras.eval(op(a_tensor, b_tensor))
# ```
# """
# macro metric(f)
#     isa(f, Expr) || error("invalid syntax; @metric must be used with a function definition")
#     if f.head === :function
#         name = f.args[1].args[1]
#         f.args[1].args[1] = :inner
#
#         code = esc(quote
#             function $(name)(_x, _y)
#                 $(f)
#                 return inner(Tensor(_x), Tensor(_y)).o
#             end
#         end)
#
#         return code
#     else
#         error("invalid syntax; @metric must be used with a function definition")
#     end
# end


export metric
