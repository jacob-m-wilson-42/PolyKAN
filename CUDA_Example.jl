# This file will show a brief example of how to leverage the GPU for a SimpleKAN architecture. CUDA must be installed and a compatible NVIDIA GPU must
# be present on the processing hardware. For more information on CUDA, see CUDA docs.

using Flux
using CUDA
CUDA.allowscalar(false) # disallow scalar operations for speed, throw an error if a scalar operation is detected
if CUDA.functional() # check to see if CUDA is running properly. You can safely comment out this CUDA-related block of code if you wish to only use the CPU
    println("CUDA is functional! Running with the GPU.")
    flush(stdout)
end

include("KAN_funcs.jl")

num_input_dimensions = 10
x = rand(num_input_dimensions,10000) # generate some input data

# create the model
model = Chain(SimpleKAN(num_input_dimensions=>10,poly_degree=4), # SimpleKAN layer
              x -> x ./ 10, # Kolmogorov compression
              SimpleKAN(10=>10,poly_degree=4),
              x -> x ./ 10,
              SimpleKAN(10=>1,poly_degree=4))


println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
println("Observe data types before loading to GPU.\n")
display(model)
display(x)
println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

println("\n\n\n")
model = gpu(model) # load modle onto gpu
x = gpu(x)

println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
println("Observe data types after loading to GPU.\n")
display(model)
display(x)
println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


println("\nModel output:")
output = model(x) # evaluate model

# Network training will occur identically to the performance demonstrations. Just be sure to keep all 
# data touching the network on the GPU. If you do not, you will get type errors and perhaps (non-bits type)
# errors. When working with the GPU in Julia, if you don't expect and don't recognize an error, it is likely
# because a GPU type touched a CPU type.

# Note that small networks (such as those shown in the demonstrations) will likely run faster on the CPU.










