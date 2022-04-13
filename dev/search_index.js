var documenterSearchIndex = {"docs":
[{"location":"home.html#Equation-Learning","page":"Home","title":"Equation Learning","text":"","category":"section"},{"location":"home.html","page":"Home","title":"Home","text":"This package allows for the learning of mechanisms T(t mathbfalpha), D(u mathbfbeta), and R(u mathbfgamma) for  delay, diffusion, and reaction model, for some cell data u and points (x t), assuming that u satisfies the model","category":"page"},{"location":"home.html","page":"Home","title":"Home","text":"fracpartial upartial t = T(t mathbfalpha)leftfracpartialpartial xleft(D(u mathbfbeta)fracpartial upartial xright) + R(u mathbfgamma)right","category":"page"},{"location":"home.html","page":"Home","title":"Home","text":"The package fits a Gaussian process to the data u at these points (x t) and uses it to draw samples from, allowing for multiple estimates of the parameters mathbfalpha, mathbfbeta, and mathbfgamma to be obtained, thus providing uncertainty quantification for these learned mechanisms. See our paper ... for more details. The main function exported by this package is bootstrap_gp which actual fits a given model with uncertainty quantification.","category":"page"},{"location":"tut.html#Tutorial","page":"Tutorial","title":"Tutorial","text":"","category":"section"},{"location":"tut.html","page":"Tutorial","title":"Tutorial","text":"Here we describe how the methods in this package can be used. We illustrate this on the 12,000 cells per well dataset from Jin et al. (2016). We only show how we could fit a delayed Fisher-Kolmogorov model. We start with the following:","category":"page"},{"location":"tut.html","page":"Tutorial","title":"Tutorial","text":"# Packages\nusing EquationLearning      # Load our actual package \nusing DelimitedFiles        # For loading the density data of Jin et al. (2016).\nusing DataFrames            # For conveniently representing the data\nusing CairoMakie            # For creating plots\nusing LaTeXStrings          # For adding LaTeX labels to plots\nusing Random                # For setting seeds \nusing LinearAlgebra         # For setting number of threads to prevent StackOverflowError\nusing Setfield              # For modifying immutable structs\n# Plots and setup\ncolors = [:black, :blue, :red, :magenta, :green]\nLinearAlgebra.BLAS.set_num_threads(1)\n# Read in the data \nfunction prepare_data(filename) # https://discourse.julialang.org/t/failed-to-precompile-csv-due-to-load-error/70146/2\n    data, header = readdlm(filename, ',', header=true)\n    df = DataFrame(data, vec(header))\n    df_new = identity.(df)\n    return df_new\nend\nassay_data = Vector{DataFrame}([])\nx_scale = 1000.0 # μm ↦ mm \nt_scale = 24.0   # hr ↦ day \nfor i = 1:6\n    file_name = string(\"data/CellDensity_\", 10 + 2 * (i - 1), \".csv\")\n    dat = prepare_data(file_name)\n    dat.Position = convert.(Float64, dat.Position)\n    dat.Time = convert.(Float64, dat.Time)\n    dat.Position ./= x_scale\n    dat.Dens1 .*= x_scale^2\n    dat.Dens2 .*= x_scale^2\n    dat.Dens3 .*= x_scale^2\n    dat.AvgDens .*= x_scale^2\n    dat.Time ./= t_scale\n    push!(assay_data, dat)\nend\nK = 1.7e-3 * x_scale^2 # Cell carrying capacity as estimated from Jin et al. (2016).\ndat = assay_data[2] # The data we will be using in this tutorial","category":"page"},{"location":"tut.html#PDE-parameters","page":"Tutorial","title":"PDE parameters","text":"","category":"section"},{"location":"tut.html","page":"Tutorial","title":"Tutorial","text":"The first step is to define the PDE setup. Our function needs a PDE_Setup struct from the following function:","category":"page"},{"location":"tut.html","page":"Tutorial","title":"Tutorial","text":"struct PDE_Setup\n    meshPoints::AbstractVector\n    LHS::Vector{Float64}\n    RHS::Vector{Float64}\n    finalTime::Float64\n    δt::AbstractVector\n    alg\nend","category":"page"},{"location":"tut.html","page":"Tutorial","title":"Tutorial","text":"The field meshPoints gives the grid points for the discretised PDE, LHS gives the coefficients in the boundary condition $a_0u(a, t) - b_0\\partial u(a, t)/\\partial x = c_0, RHS gives the coefficients in the boundary condition a_1u(b, t) + b_1\\partial u(b, t)/\\partial x = c_1$, finalTime gives the time that the solution is solved up to, δt gives the vector of points to return the solution at, and alg gives the algorithm to use for solving the system of ODEs arising from the discretised PDE. We use the following:","category":"page"},{"location":"tut.html","page":"Tutorial","title":"Tutorial","text":"δt = LinRange(0.0, 48.0 / t_scale, 5)\nfinalTime = 48.0 / t_scale\nN = 1000\nLHS = [0.0, 1.0, 0.0]\nRHS = [0.0, -1.0, 0.0]\nalg = Tsit5()\nmeshPoints = LinRange(75.0 / x_scale, 1875.0 / x_scale, 500)\npde_setup = PDE_Setup(meshPoints, LHS, RHS, finalTime, δt, alg)","category":"page"},{"location":"tut.html","page":"Tutorial","title":"Tutorial","text":"Note that these boundary conditions LHS and RHS correspond to no flux boundary conditions.","category":"page"},{"location":"paper.html#VandenHeuvel-et-al.-(2022)","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"","category":"section"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"This section briefly discusses our paper, and steps for reproducing the figures in the paper. The paper  can be found here ..., and the abstract of the paper is:","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"Parameter estimation for biological processes is often a difficult problem and depends significantly on the quality and quantity of avaiable data. We introduce a new framework which utilises Gaussian processes to discover the mechanisms underlying delay, diffusion, and reaction in a cell invasion process. Gaussian processes are leveraged with bootstrapping to provide uncertainty quantification for the mechanisms that drive the invasion process. Our framework is efficient and easily parallelisable, and can be applied to other problems. We illustrate our methods on a scratch assay experiment, demonstrating how simply we can explore different functional forms and develop and test hypotheses about underlying mechanisms, such as whether delay is present in the cell invasion process.","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"The scratch assay data from Jin et al. (2016) can be found in this GitHub repository in VandenHeuvel2022_PaperCode/data.","category":"page"},{"location":"paper.html#Paper-results","page":"VandenHeuvel et al. (2022)","title":"Paper results","text":"","category":"section"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"The main body of the paper is produced using the code in VandenHeuvel2022PaperCode/papercode.jl. Below, we list the sections that this script is broken into, along with descriptions of these sections:","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"Load the required packages","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"Here we simply load all the necessary packages.","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"Set some global parameters ","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"(Note that we also have a section called this in 7. below.) This section defines some parameters for plotting that are used in most of the sections. We also write:","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"LinearAlgebra.BLAS.set_num_threads(1)","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"This setting was used to remove issues relating to A \\ b giving StackOverflowError. See, for example, #43301 or #43242.","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"Read in the data from Jin et al. (2016)","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"Here we read in the data from Jin et al. (2016), scaling the data by hat x = 1000 and hat t = 24.","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"Figure X: Plotting the density data from Jin et al. (2016)","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"This code plots the data from Jin et al. (2016) and also plots a curve through the average of the experimental replicates at each point in sspace and time.","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"Figure X: Plotting the Gaussian processes fit to the data from Jin et al. (2016)","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"This section plots the Gaussian processes over the data from Jin et al. (2016), These Gaussian processes are fit using GaussianProcesses.jl.","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"Figure X: Plotting the space-time diagram for the Gaussian process","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"This section plots the same Gaussian processes, but now plots them on the (x t) place, colouring the points by the mean of the Gaussian process posterior at each point (x t).","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"Set some global parameters ","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"This section now defines the global parameters for the bootstrapping. The parameters for the PDE are defined first, and then the parameters for bootstrapping. We also remove the left-most points from the data from Jin et al. (2016) here.","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"Model fits","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"This section contains the actual code that gives the figures in the paper. There are five functions that we define first:","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"model_fits: This function fits, for a given dataset, a Fisher-Kolmogorov model (with and without delay), a Porous-Fisher model (with and without delay), and a delayed generalised Porous-FKPP model.\nplot_fisher_kolmogorov_delay: This function plots the results from a delayed Fisher-Kolmogorov model.\nplot_generalised_fkpp_delay: This function plots the results from a delayed generalised Porous-FKPP model.\nplot_pde_soln!: This function adds, to an existing figure, an axis for the PDE solutions for a given dataset.\nplot_pde_soln: For the six datasets, this function plots all of the PDE solutions from each dataset on the same figure.","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"After these functions are defined, we define parameters that scale each parameter for each function such that the scaled parameters that we have to estimate are all mathcal O(1). We based these parameter scales on the results from Jin et al. (2016) and Lagergren et al. (2020), or adjusted further based on issues we observed when fitting models. We then fit all the models, which takes a reasonably long time to complete. We then make all the plots.","category":"page"},{"location":"paper.html#Simulation-studies","page":"VandenHeuvel et al. (2022)","title":"Simulation studies","text":"","category":"section"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"We also present several simulation studies in the paper, all of which are given in VandenHeuvel2022PaperCode/simulationstudies.jl.","category":"page"},{"location":"paper.html#Simulation-study-I:-Fisher-Kolmogorov-Model,-10,000-cells-per-well","page":"VandenHeuvel et al. (2022)","title":"Simulation study I: Fisher-Kolmogorov Model, 10,000 cells per well","text":"","category":"section"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"In this study we fit some models to data simulated from the Fisher-Kolmogorov model","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"fracpartial upartial t = beta_1fracpartial^2upartial x^2 + gamma_1uleft(1-fracuKright)","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"Running all this code will produce the figures in the corresponding section of our paper.","category":"page"},{"location":"paper.html#Simulation-study-I:-Fisher-Kolmogorov-Model,-10,000-cells-per-well-2","page":"VandenHeuvel et al. (2022)","title":"Simulation study I: Fisher-Kolmogorov Model, 10,000 cells per well","text":"","category":"section"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"In this study we fit some models to data simulated from the Fisher-Kolmogorov model","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"fracpartial upartial t = beta_1fracpartial^2upartial x^2 + gamma_1uleft(1-fracuKright)","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"Running all this code will produce the figures in the corresponding section of our paper.","category":"page"},{"location":"paper.html#Simulation-study-II:-Fisher-Kolmogorov-Model-with-delay,-10,000-cells-per-well","page":"VandenHeuvel et al. (2022)","title":"Simulation study II: Fisher-Kolmogorov Model with delay, 10,000 cells per well","text":"","category":"section"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"In this study we fit some models to data simulated from the delayed Fisher-Kolmogorov model","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"fracpartial upartial t = frac11+exp(-alpha_1-alpha_2t)leftbeta_1fracpartial^2upartial x^2 + gamma_1uleft(1-fracuKright)right","category":"page"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"Running all this code will produce the figures in the corresponding section of our paper.","category":"page"},{"location":"paper.html#Simulation-study-III:-Fisher-Kolmogorov-model,-10,000-cells-per-well,-basis-function-approach","page":"VandenHeuvel et al. (2022)","title":"Simulation study III: Fisher Kolmogorov model, 10,000 cells per well, basis function approach","text":"","category":"section"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"This study fits the same model as in study I, but using the basis function approach. Running all this code will produce the figures in the corresponding section of our paper.","category":"page"},{"location":"paper.html#Simulation-study-IV:-Data-thresholding-on-the-Fisher-Kolmogorov-model-of-Study-I","page":"VandenHeuvel et al. (2022)","title":"Simulation study IV: Data thresholding on the Fisher-Kolmogorov model of Study I","text":"","category":"section"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"This study considers the effects of data thresholding on the model in the first study. This study is done by simply looping over many tolerance values. Running all this code will produce the figures in the corresponding section of our paper.","category":"page"},{"location":"paper.html#Simulation-study-V:-Data-thresholding-on-the-Fisher-Kolmogorov-model-of-Study-II","page":"VandenHeuvel et al. (2022)","title":"Simulation study V: Data thresholding on the Fisher-Kolmogorov model of Study II","text":"","category":"section"},{"location":"paper.html","page":"VandenHeuvel et al. (2022)","title":"VandenHeuvel et al. (2022)","text":"This study considers the effects of data thresholding on the model in the second study. This study is done by simply looping over many tolerance values. Running all this code will produce the figures in the corresponding section of our paper.","category":"page"}]
}
