using Flux
using Plots
using Symbolics
using Latexify
using Distributions

include("KAN_funcs.jl")
function add_latex_newline(latex_string,num_terms)
    # adds a newline to a function result output by Latexify. Currently only supports a single equation (no matrix)
    # num_terms - integer number of terms place before the newline
    # latex_string - output of latexify() string
    extracted_func = split(latex_string,"\n")[4] # grab the function from the latex string
    poly_terms = split(extracted_func,r"(?=[+-])(?=(?:[^{}]*\{[^{}]*\})*[^{}]*$)") # split by polynomial terms
    modified_string = ""
    for i in 1:length(poly_terms)
        if i%num_terms == 0
            modified_string = modified_string * poly_terms[i] * "\\\\" # append \\ (not \\\\) to every num_terms
        else
            modified_string = modified_string * poly_terms[i] * "&" # continually stack string together and add line breaks
        end
    end
    outstring = replace(latex_string,extracted_func=>modified_string)
    outstring = replace(outstring,"\\begin{array}{c}"=>"\\begin{array}{$(join(fill("l",num_terms)))}")
    return outstring

end

# function to define training loop. It is best to place compuationally intensive code inside functions to help the Julia compiler (avoid global variables)
function begintraining()
    plots = []
    fig1=fig2=[]

    # define the model
    model = Chain(SimpleKAN(1=>4,poly_degree=4),
                temp -> temp ./ 10,
                SimpleKAN(4=>4,poly_degree=4),
                temp -> temp ./ 10,
                SimpleKAN(4=>1,poly_degree=4)) 

                
    # set up optimizer with default learning rate            
    optim = Flux.setup(Flux.Optimiser(Adam(1f-3)),model) 

    # define target function to regress towards
    target_func(input) = sinc.(input)

    # generate some input and target data
    maxval = 2f0 * pi
    x = -maxval .+ (maxval - (-maxval)) .* rand(Float32,1,1000) # draw from +/- maxval uniformly
    targets = target_func(x)

    # normalize input for use in network
    x_normalized = x ./ maxval

    error = 1f10 # initialize error to a large value to start
    err_progress = Float32[]
    num_epochs = 100000 # select number of epochs to train for

    plotx = -2f0*pi:0.01f0:2f0*pi
    modelplotx = reshape(collect(plotx),1,:) ./ maxval
    
    start_time = time()
    for epoch_num in 1:num_epochs
        (error,grads) = Flux.withgradient(tempmodel -> Flux.mse(tempmodel(x_normalized),targets),model)
        push!(err_progress,error)
        println("Epoch Number: $epoch_num of $num_epochs, Error: $error") # print progress to the screen
        Flux.update!(optim, model, grads[1]) # update the model using the optimizer

        # plot progress live
        if epoch_num % 100 == 1 # plot results every 100 epochs
            fig1 = plot(plotx,target_func(plotx),xlabel="Angle (rad)",ylabel="Value",title="Sinc",label="Target",dpi=600)
            plot!(fig1,plotx,model(modelplotx)[:],label="Predicted")

            numepochs2plot = 10000
            if epoch_num < numepochs2plot
                fig2 = plot(1:epoch_num,err_progress,xlabel="Epoch",ylabel="Error",label="Error")
            else
                fig2 = plot(epoch_num-numepochs2plot:epoch_num,err_progress[epoch_num-numepochs2plot:end],xlabel="Epoch",ylabel="Error",label="Error")
            end
            plot(fig1,fig2) |> display

            push!(plots,fig1)
            

        end
    end
    elapsed_time = round(time() - start_time,digits=2)
    println("Training ended. Time elapsed: $elapsed_time s.")

    # generate symbolic result
    @variables x y # generate symbolic variables, an array of automatically named (numbered) vars is possible. See Symbolics docs for more information
    symbolic_input = reshape([x,y],(2,1)) # reshape into 2x1 matrix that is expected by the model
    symbolic_trace_noexpand = model(symbolic_input) # trace out the symboic form of the model
    symbolic_trace = simplify.(symbolic_trace_noexpand,expand=true) # simplify the symbolic result (expand=true combines all polynomial degrees)
    println("Simplified symbolic result:\n$(symbolic_trace[1])") # print results to screen
    latex_trace = latexify(symbolic_trace) # generate latex result for use in documentation/publication
    latex_trace = add_latex_newline(latex_trace,4) # add newlines for easier readablility

    # compare number of parameters of simplified expression and full model
    num_simplified_params_expand = length(collect(eachmatch(r"[-+]?\d*\.\d+([eE][-+]?\d+)?",string(symbolic_trace[1])))) # match all floating point values and scientific notation numbers. Ignore integers
    num_simplified_params_noexpand = length(collect(eachmatch(r"[-+]?\d*\.\d+([eE][-+]?\d+)?",string(symbolic_trace_noexpand[1])))) # match all floating point values and scientific notation numbers. Ignore integers
    num_original_params = sum(length.(Flux.params(model)))
    println("Original number of model parameters: $num_original_params\nNumber of parameters after simplification and expansion: $num_simplified_params_expand\nNumber of parameters after simplification without expansion: $num_simplified_params_noexpand")



    ##############
    # save results 
    ##############
    # plot and save results
    fig2save = plot(plotx,target_func(plotx),xlabel="Angle (rad)",ylabel="Value",title="Sinc",label="Target",dpi=600,legend=true)
    plot!(fig2save,plotx,model(modelplotx)[:],label="Predicted",dpi=600,legend=true)
    savefig(fig2save,"Performance_Demonstration_Figures/Demonstration_Sinc.png")

    # save latex symbolic output to file
    open("Performance_Demonstration_Symbolic_Results/Demonstration_Sinc_Result.txt", "w") do file
        write(file, "Latex form of symbolic result:\n\n")
        write(file, latex_trace)
        write(file,"\nOriginal number of model parameters: $num_original_params\nNumber of parameters after simplification and expansion: $num_simplified_params_expand\nNumber of parameters after simplification without expansion: $num_simplified_params_noexpand")
    end
    return nothing
end
begintraining() # start training

