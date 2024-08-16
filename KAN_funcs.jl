# This file is used to define KAN-specific functions for use. If you simply copy this file into your project,
# you can access all KAN-specific functions described in the paper.

# define a stucture (type) that defines what data a PolyKAN can hold
mutable struct PolyKAN{N,M}
    num_input_nodes::N
    num_output_nodes::N
    poly_degree::N
    weights::M
end

# functor marks all parameters for training. By default, flux ignores scalar values so only the weights are trainable. 
# This can be checked by calling Flux.params() on an initialized model to return all trainable parameters that
# Flux can see
Flux.@functor PolyKAN 


# These functions return an Mxp+1 matrix of evaluated polynomial values i.e. f(x) = 1 + x + x^2; f(2) = 1 + 2 + 2^2
# Data is oriented column major. A cpu and GPU format is created. The specific version that gets called is dependent
# on the data types of the input
# x - input, column major matrices for multiple data points
# p - desired polynomial degree
function build_poly_mat(x::Union{Vector,Real},p::Int) # CPU version
        return x'.^ collect(0:p)
 end
 function build_poly_mat(x::AbstractVector,p::Int) # GPU version
        return x'.^ CuArray(collect(0:p)) # the collect call is required to avoid llvm compile error
 end

# Define PolyKAN constructor. This function helps build an object of the previously 
# defined PolyKAN type.
# Example usage: model = PolyKAN(3=>5,3) -> define architecture with input 
# dimensionality of 3 and 5 neurons per input. Use polynomial degree 3.
# To evaluate an output use model(x).
function PolyKAN(node_config::Pair;poly_degree::Int=5,init=rand)
    input_nodes = node_config[1]
    output_nodes = node_config[2]
    weights = [init(Float32,output_nodes,poly_degree+1) for i in 1:input_nodes]
    return PolyKAN(input_nodes,output_nodes,poly_degree,weights)
end

# define how the output of a layer is computed
function (l::PolyKAN)(data::AbstractMatrix)
    return sum([l.weights[innode] * build_poly_mat(data[innode,:],l.poly_degree) for innode in 1:l.num_input_nodes])
end






