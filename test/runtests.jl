using Keras
using StatsBase

using Base.Test

import Keras.Layers: Dense, Activation, LSTM

mse(actual, pred) = mean(square(actual - pred))
mae(actual, pred) = mean(abs(actual - pred))
rmse(actual, pred) = sqrt(mse(actual, pred))

@testset "Keras" begin
    @testset "Basic Usage" begin
        model = Sequential()
        add!(model, Dense(80, input_dim=30))
        add!(model, Activation(:relu))
        add!(model, Dense(10))
        add!(model, Activation(:softmax))
        compile!(
            model;
            loss=:categorical_crossentropy,
            optimizer=:sgd,
            metrics=[:accuracy]
        )

        h = fit!(model, rand(100, 30), rand(100, 10); epochs=20, batch_size=10, verbose=0)

        @test haskey(h[:history], "acc")
        @test haskey(h[:history], "loss")

        evaluate(model, rand(10, 30), rand(10, 10); batch_size=5, verbose=0)
        predict(model, rand(10, 30); batch_size=5, verbose=0)
    end

    @testset "Custom Objectives & Metrics" begin
        model = Sequential()
        add!(model, Dense(20, input_dim=30))
        add!(model, Activation(:relu))
        add!(model, Dense(10))
        add!(model, Activation(:softmax))

        compile!(
            model;
            loss=metric(mse),
            optimizer=:sgd,
            metrics=[:accuracy, metric(mae), metric(rmse)]
        )

        h = fit!(model, rand(100, 30), rand(100, 10); epochs=10, batch_size=10, verbose=0)

        @test haskey(h[:history], "acc")
        @test haskey(h[:history], "loss")
        @test haskey(h[:history], "mae")
        @test haskey(h[:history], "rmse")

        evaluate(model, rand(10, 30), rand(10, 10); batch_size=5, verbose=0)
        predict(model, rand(10, 30); batch_size=5, verbose=0)
    end

    @testset "Tensor Operations" begin
        @testset "Single Tensor Operations" begin
            x = rand(Float32, 4, 3)
            x_t = variable(x)

            @testset "Testing $op" for op in [-, transpose, sqrt, exp, log, round, sin, cos]
                expected = op(x)
                result = Keras.eval(op(x_t))

                @test size(expected) == size(result)
                @test typeof(expected) == typeof(result)
                @test all(map(isapprox, expected, result))
            end

            @testset "Testing $op" for op in [maximum, minimum, sum, prod, var, std, mean]
                expected = Float32(op(x))
                result = Float32(Keras.eval(op(x_t))[1])
                @test_approx_eq_eps expected result 0.1
            end

            x = rand(Bool, 4)
            x_t = variable(x)
            @testset "Testing $op" for op in [all, any]
                expected = op(x)
                result = Bool(Keras.eval(op(x)))
                @test expected == result
            end
        end

        @testset "Two Tensor Operations" begin
            a = rand(Float32, 4, 3)
            a_t = variable(a)

            b = rand(Float32, 4, 3)
            b_t = variable(b)

            @testset "$op" for op in [.==, .!=, .>, .<, .>=, .<=]
                expected = op(a, b)
                result = BitArray(Keras.eval(op(a_t, b_t)))
                @test expected == result
            end

            @testset "Testing $op" for op in [mod, .^, .-, .+, .*, ./]
                expected = op(a, b)
                result = Keras.eval(op(a_t, b_t))
                @test size(expected) == size(result)
                @test typeof(expected) == typeof(result)
                @test all(map(isapprox, expected, result))
            end

            b = rand(Float32, 3, 4)
            b_t = variable(b)

            @testset "Testing $op" for op in [*]
                expected = op(a, b)
                result = Keras.eval(op(a_t, b_t))

                @test size(expected) == size(result)
                @test typeof(expected) == typeof(result)
                @test all(map(isapprox, expected, result))
            end
        end

        @testset "Custom Tensor Operations" begin
            x = Float32[-1.1, 0.8, -0.5, 1.2]
            x_t = variable(x)

            @test Keras.eval(clip(x_t, -1.0, 1.0)) == Float32[-1.0, 0.8, -0.5, 1.0]
            square(x_t)
            Keras.cast(x_t, :float64)
        end
    end

    @testset "Layers" begin
        @testset "Basic Usage" begin
            a = Keras._layers[:Input](shape=(140, 256))

            lstm = LSTM(32)
            encoded_a = lstm(a)

            W = weights(lstm)
            weights!(lstm, W)
            config(lstm)
            input(lstm)
            output(lstm)
            input_shape(lstm)
            output_shape(lstm)
        end
    end
end
