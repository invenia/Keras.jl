using Conda
using PyCall

backend = "theano"

try
    @pyimport keras
    if VersionNumber(keras.__version__) >= v"2.0.2"
        info("Using Keras $(keras.__version__) -> $(keras.__path__)")
        backend = keras.backend[:backend]()
    else
        error("Invalid Keras version ($(keras.__version__))")
    end
catch _
    info("No valid (>=2.0.2) install of keras found.")
    info("Installing Keras via Conda.jl...")
    Conda.add_channel("conda-forge")
    Conda.add("keras==2.0.2")

    if is_windows()
        warn("Tensorflow only supports python 3.5 on windows. Using the included theano backend.")
    else
        info("Installing Tensorflow (1.0.0) via Conda.jl")
        Conda.add("tensorflow==1.0.0")
        backend = "tensorflow"
    end
end

keras_path = joinpath(homedir(), ".keras")
config_path = joinpath(keras_path, "keras.json")
default_settings = """
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "$backend"
}
"""

if !ispath(config_path)
    if !ispath(keras_path)
        mkdir(keras_path)
    end

    info("Writing default config to $config_path")

    open(config_path, "w+") do fstream
        write(fstream, default_settings)
    end
end
