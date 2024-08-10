# SimpleKAN: A Polynomial Approach to Kolmogorov-Arnold Networks with Symbolic Computation
This repository contains files to replicate all results of the SimpleKAN paper (submission in progress). Below is a brief description of the included files. All demonstration files are designed to be run independently with functions supplied from KAN_funcs.jl.

# Getting Started
We will assume the user is using VSCode as the Julia editor which can be installed [here](https://code.visualstudio.com/download). We will also assume the user is running Julia version 1.10.0 , however, future versions of Julia below 2.0.0 should remain compatible. Julia can be installed [here][https://julialang.org/downloads/]. A great introductory tutorial for installing and using Julia in VSCode can be found [here](https://code.visualstudio.com/docs/languages/julia).

## Brief GIT Tutorial and Code Cloning
This repository can be cloned using git. First install [git](https://git-scm.com/downloads). Then click the code dropdown button
![image](https://github.com/user-attachments/assets/76521580-f272-418b-93b8-43127374ef7d)

and copy the HTTPS link. Navigate to the directory in which you would like to clone the repository from the VSCode Powershell interface, the Windows CMD (Mac Terminal) or other shell application. Type the below code to clone the repository. Note, first time git users may be prompted to sign in and establish other credentials. A more detailed git tutorial can be found [here](https://git-scm.com/docs/gittutorial).
```
git clone [url]
```



STILL IN PROGRESS...

## Brief File Description
### KAN_funcs.jl
This file contains all files necessary to create SimpleKAN models and compute outputs. It can be copied into a user's project directory and included into a file using the include() function. It will give the user the ability to create and use SimpleKAN models that are compatible with the Flux package.

### Performance Demonstration Files
These files can be directly run without modification to reproduce the results in the paper. 


