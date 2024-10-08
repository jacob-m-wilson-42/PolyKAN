# <p align="center"> PolyKAN: A Polynomial Approach to Kolmogorov-Arnold Networks with Symbolic Computation
This repository contains files to replicate all results of the PolyKAN paper (pending review). Below is a brief description of the included files. All demonstration files are designed to be run independently with functions supplied from KAN_funcs.jl.

<p align="center"><img src="https://github.com/user-attachments/assets/a683de0e-38b2-4c3d-a286-b93223529541" width="750">

# Getting Started
We will assume the user is using VSCode as the Julia editor which can be installed [here](https://code.visualstudio.com/download). We will also assume the user is running Julia version 1.10.0 , however, future versions of Julia below 2.0.0 should remain compatible. Julia can be installed [here](https://julialang.org/downloads/). Also ensure the Julia extension in VSCode is installed. A great introductory tutorial for installing and using Julia in VSCode can be found [here](https://code.visualstudio.com/docs/languages/julia).

## Brief GIT Tutorial and Code Cloning
This repository can be cloned using git. First install [git](https://git-scm.com/downloads). Then click the code dropdown button
![image](https://github.com/user-attachments/assets/3459b3f4-220a-4f3d-baa9-158bdd3b0a37)

and copy the HTTPS link. Navigate to the directory in which you would like to clone the repository from the VSCode Powershell interface, the Windows CMD (Mac Terminal) or other shell application. Type the below code to clone the repository. Note, first time git users may be prompted to sign in and establish other credentials. A more detailed git tutorial can be found [here](https://git-scm.com/docs/gittutorial).
```
git clone [url]
```

## Julia Package Installation
Julia provides commands to simply install all required dependencies of a cloned repository using only a few lines of code. The versions of each package are kept consistent to avoid versioning errors. We created this file using Julia version 1.10.0. Near future versions should remain compatible but may introduce versioning issues within the installed packages.

Begin by navigating to the directory in which you cloned the repository. Ensure you are inside of the cloned repository and not one directory out of it. Run Julia by typing `Julia`. You should see the Julia REPL launch.

![image](https://github.com/user-attachments/assets/9dc474ca-fa16-420a-8495-aadfd6dce5dd)

Press the `[` key. You should see the prompt change to something similar to the below.

```
(@v1.10) pkg>
```

Now type

```
(@v1.10) pkg> activate .
```
Be sure to not forget the "." . You should now notice the REPL prompt has changed to reflect the new Julia environment you have just entered. In this case the REPL indicates we are in the environment `(PolyKAN)`. Different environments can be used for different projects to ensure various packages do not conflict. Because you are now in a new environment which has no packages at the moment, we have to install all required packages in the same versions as the repository. We can simply do this by typing

```
(PolyKAN) pkg> instantiate
```

This should install all dependencies for you! Note that VSCode should create a .vscode folder which will contain information specific to the environment you are working with. In general, when restarting VSCode, ensure the bottom of the window displays the correct working environment and **NOT** the default environment that is shown below.

![image](https://github.com/user-attachments/assets/aca78ff8-309e-46c1-8cce-2c989bf5627a)

More detailed information on package installation can be found [here](https://pkgdocs.julialang.org/v1/environments/). Note to leave the REPL package manager, simply type the backspace key. To run files in VSCode, simply open a file and press the run key in the top right of the window (by default).

![image](https://github.com/user-attachments/assets/7b5bd035-ef6b-4a5c-9fc1-6af236c5893d)

Simple as that! We provided three files for users to experiment with and replicate results from the paper. We additionally included a file that describes how to use CUDA to run our network on the GPU!


# Brief File Description
## KAN_funcs.jl
This file contains all files necessary to create PolyKAN models and compute outputs. It can be copied into a user's project directory and included into a file using the `include()` function. It will give the user the ability to create and use PolyKAN models that are compatible with the Flux package.

## Performance Demonstration Files
These files can be directly executed without modification to reproduce the results in the paper. 

### CUDA_Example.jl
This file contains information about how to utilize the GPU with our network!

# Citation
If you use our work in your research, please cite the following paper:
```
@article{PolyKAN,
  author  = "Wilson, Jacob and Voelz, David",
  title   = "PolyKAN: A Polynomial Approach to Kolmogorov-Arnold Networks with Symbolic Computation",
  journal = "Journal of Artificial Intelligence Research (JAIR) (Submission Pending)",
  year    = "2024",
}
```


