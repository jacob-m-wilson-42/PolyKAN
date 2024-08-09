using Flux
using Plots
using Symbolics
using Latexify
using SpecialFunctions

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
    num_epochs = 100000 # select number of epochs to train for

    fig1=fig2=fig3=[] # initialize variables for use in plotting

    # define the model
    model = Chain(SimpleKAN(2=>4,poly_degree=4),
                temp -> temp ./ 10,
                SimpleKAN(4=>1,poly_degree=4)) 

                
    # set up optimizer with default learning rate            
    optim = Flux.setup(Flux.Optimiser(Adam(1f-3)),model) 

    # define target function to regress towards
    target_func(input) = map(temp->erf(temp[1],temp[2]),eachcol(input))

    # generate some input and target data
    maxval = 5
    x = -maxval .+ (maxval - (-maxval)) .* rand(Float32,2,1000) # draw from +/- maxval uniformly
    targets = target_func(x) # determine output values for the random samples

    # normalize input for use in network
    x_normalized = x ./ maxval

    error = 1f10 # initialize error to a large value to start
    err_progress = Float32[] # vector to store error progress

    # generate sample points for plotting
    plotx = range(-maxval,maxval,100)
    ploty = plotx
    modelplotx = plotx .* ones(1,100) # use Julia's mapping operation to generate a matrix similar to Matlab's meshgrid
    modelploty = modelplotx' # transpose to generate y coords
    modelplot = vcat(modelplotx[:]',modelploty[:]') # stretch the two matrices into vectors and stack the vectors into a 2xN matrix (the form the model is expecting)
    modelplot_normalized = modelplot ./ maxval # normalize input data for use in the model

    
    # for the requested number of epochs
    start_time = time()
    for epoch_num in 1:num_epochs
        (error,grads) = Flux.withgradient(tempmodel -> Flux.mse(tempmodel(x_normalized)[:],targets),model) # calculate the error
        push!(err_progress,error) # store the error for this epoch
        println("Epoch Number: $epoch_num of $num_epochs, Error: $error") # print progress to the screen
        Flux.update!(optim, model, grads[1]) # update the model using the optimizer

        # plot progress live
        if epoch_num % 100 == 1 # plot results every 100 epochs

            # generate images to plot
            predicted_image = reshape(model(modelplot_normalized),size(modelplotx))
            target_image = reshape(target_func(modelplot),size(modelplotx))
            error_image = target_image .- predicted_image

            # generate plots but do not display just yet
            fig1 = heatmap(plotx,ploty,predicted_image,title="Predicted")
            fig2 = heatmap(plotx,ploty,target_image,title="Target") 
            fig3 = heatmap(plotx,ploty,error_image,title="Error")

            # plot training error curve and some logic to zoom into the last numepochs2plot
            numepochs2plot = 10000
            if epoch_num < numepochs2plot
                fig4 = plot(1:epoch_num,err_progress,xlabel="Epoch",ylabel="Error",label="Error")
            else
                fig4 = plot(epoch_num-numepochs2plot:epoch_num,err_progress[epoch_num-numepochs2plot:end],xlabel="Epoch",ylabel="Error",label="Error")
            end

            # display the generated plots
            plot(fig1,fig2,fig3,fig4) |> display
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
    # generate final matrices to plot
    predicted_image = reshape(model(modelplot_normalized),size(modelplotx))
    target_image = reshape(target_func(modelplot),size(modelplotx))
    error_image = target_image .- predicted_image

    # plot and save results
    fig2save_pred = heatmap(plotx,ploty,predicted_image,dpi=600,xlabel="x",ylabel="y",title="Predicted")
    fig2save_tar = heatmap(plotx,ploty,target_image,dpi=600,xlabel="x",ylabel="y",title="Target")
    fig2save_err = heatmap(plotx,ploty,error_image,dpi=600,xlabel="x",ylabel="y",title="Error")
    fig2save_errcurve = plot(1:num_epochs,err_progress,xlabel="Epoch",ylabel="Error",legend=false,title="Error")
    savefig(fig2save_pred,"Performance_Demonstration_Figures/Demonstration_erf_Predicted.png")
    savefig(fig2save_tar,"Performance_Demonstration_Figures/Demonstration_erf_Target.png")
    savefig(fig2save_err,"Performance_Demonstration_Figures/Demonstration_erf_Error_Image.png")
    savefig(fig2save_errcurve,"Performance_Demonstration_Figures/Demonstration_erf_Error_Curve.png")

    # save latex symbolic output to file
    open("Performance_Demonstration_Symbolic_Results/Demonstration_erf_Result.txt", "w") do file
        write(file, "Latex form of symbolic result:\n\n")
        write(file, latex_trace)
        write(file,"\nOriginal number of model parameters: $num_original_params\nNumber of parameters after simplification and expansion: $num_simplified_params_expand\nNumber of parameters after simplification without expansion: $num_simplified_params_noexpand")
    end
    return nothing
end
begintraining() # start training

