using Conda

Conda.add_channel("conda-forge")
Conda.add("tensorflow=0.9.0")
Conda.add("keras==1.0.7")

keras_path = joinpath(homedir(), ".keras")
config_path = joinpath(keras_path, "keras.json")
default_settings = """
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
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
