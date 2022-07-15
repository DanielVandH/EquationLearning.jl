## Data and packages
using DataFrames
using DelimitedFiles
using CairoMakie
using Random
using Distributions
fontsize = 20

function prepare_data(filename) # https://discourse.julialang.org/t/failed-to-precompile-csv-due-to-load-error/70146/2
    data, header = readdlm(filename, ',', header=true)
    df = DataFrame(data, vec(header))
    df_new = identity.(df)
    return df_new
end

assay_data = Vector{DataFrame}([])
for i = 1:6
    file_name = string("data/CellDensity_", 10 + 2 * (i - 1), ".csv")
    dat = prepare_data(file_name)
    push!(assay_data, dat)
end
K = 1.7e-3

## First figure: The entire plate 
Random.seed!(29291)
R = 9000 / 2
num_points = 5000
scratch = 750 / 2
r = R * sqrt.(rand(num_points))
θ = 2π * rand(num_points)
x = r .* cos.(θ)
y = r .* sin.(θ)
scratch_idx = abs.(x) .< scratch
window_x = 1900 / 2
window_y = 1410 / 2
x = x[.!scratch_idx]
y = y[.!scratch_idx]

fig = Figure()
ax = Axis(fig[1, 1], title="(a): Well", titlealign=:left, aspect = 1)
scatter!(ax, x, y, color=:red, markersize=4)
lines!(ax, [-window_x, window_x, window_x, -window_x, -window_x],
    [-window_y, -window_y, window_y, window_y, -window_y],
    color=:blue, linewidth=4)
hidedecorations!(ax)
save("experiment_figures/FullView_1.pdf", fig, px_per_unit=2)
save("experiment_figures/FullView_1.eps", fig, px_per_unit=2)
save("experiment_figures/FullView_1.png", fig, px_per_unit=2)

## Second figure: Initial cells per well 
num_cells = Array{Int64}(zeros(38, 3, 6, 2))
for i in 1:length(assay_data)
    num_cells[:, 1, i, 1] .= assay_data[i][:, :Num_Cells1][assay_data[i][:, :Time].==0.0] .>> 1
    num_cells[:, 2, i, 1] .= assay_data[i][:, :Num_Cells2][assay_data[i][:, :Time].==0.0] .>> 1
    num_cells[:, 3, i, 1] .= assay_data[i][:, :Num_Cells3][assay_data[i][:, :Time].==0.0] .>> 1
    num_cells[:, 1, i, 2] .= assay_data[i][:, :Num_Cells1][assay_data[i][:, :Time].==48.0] .>> 1
    num_cells[:, 2, i, 2] .= assay_data[i][:, :Num_Cells2][assay_data[i][:, :Time].==48.0] .>> 1
    num_cells[:, 3, i, 2] .= assay_data[i][:, :Num_Cells3][assay_data[i][:, :Time].==48.0] .>> 1
end
figures2 = Array{Figure}(undef, 2, 3)
plot_pos = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
colors = [:blue, :black, :red]
data_set = 4
measuring_set = 1
measuring_K_set = 6
data_set_x_1 = Vector{Vector{Float64}}([])
data_set_y_1 = Vector{Vector{Float64}}([])
data_set_x_2 = Vector{Vector{Float64}}([])
data_set_y_2 = Vector{Vector{Float64}}([])
measuring_x_1 = Vector{Vector{Float64}}([])
measuring_y_1 = Vector{Vector{Float64}}([])
measuring_x_2 = Vector{Vector{Float64}}([])
measuring_y_2 = Vector{Vector{Float64}}([])
measuring_x_K = Vector{Vector{Float64}}([])
measuring_y_K = Vector{Vector{Float64}}([])
CPW = ["(b): 10,000 cells per well", "(c): 12,000 cells per well", 
    "(d): 14,000 cells per well", "(e): 16,000 cells per well", 
    "(f): 18,000 cells per well", "(g): 20,000 cells per well"]
seeds = 1:1368
for ℓ in 1:2
    seed_idx = 1
    for j in 1:3
        fig = Figure(fontsize=fontsize)
        idx = 1
        for i in 1:length(assay_data)
            x = Vector{Float64}([])
            y = Vector{Float64}([])
            for k in 1:38
                Random.seed!(seeds[seed_idx])
                seed_idx += 1
                position_range = assay_data[i][:, :Position][k] .+ (-25, 25)
                x_sample = Uniform(position_range...)
                y_sample = Uniform(0, 2window_y)
                append!(x, rand(x_sample, num_cells[k, j, i, ℓ]))
                append!(y, rand(y_sample, num_cells[k, j, i, ℓ]))
            end
            if i == data_set
                if ℓ == 1
                    push!(data_set_x_1, x)
                    push!(data_set_y_1, y)
                elseif ℓ == 2
                    push!(data_set_x_2, x)
                    push!(data_set_y_2, y)
                end
            end
            if i == measuring_set
                if ℓ == 1
                    push!(measuring_x_1, x)
                    push!(measuring_y_1, y)
                elseif ℓ == 2
                    push!(measuring_x_2, x)
                    push!(measuring_y_2, y)
                end
            end
            if i == measuring_K_set && ℓ == 2
                push!(measuring_x_K, x)
                push!(measuring_y_K, y)
            end
            ax = Axis(fig[plot_pos[idx][1], plot_pos[idx][2]],
                title=CPW[idx],
                titlealign=:left)
            scatter!(ax, x, y, markersize=2, color=colors[j])
            xlims!(ax, 0, maximum(x))
            ylims!(ax, 0, maximum(y))
            ax.xticks = 25:350:1875
            hideydecorations!(ax)
            hidexdecorations!(ax, label=false, ticklabels=false)
            idx += 1
        end
        save("experiment_figures/WellView_$(j)_$(ℓ).pdf", fig, px_per_unit=2)
        save("experiment_figures/WellView_$(j)_$(ℓ).eps", fig, px_per_unit=2)
        save("experiment_figures/WellView_$(j)_$(ℓ).png", fig, px_per_unit=2)
        figures2[ℓ, j] = fig
    end
end

## Third figure: Stacking figures
fig_1 = Vector{Figure}(undef, 3)
for i in 1:3
    fig = Figure(fontsize=fontsize)
    ax = Axis(fig[1, 1])
    scatter!(ax, data_set_x_1[i], data_set_y_1[i], color=colors[i], markersize=4)
    hidedecorations!(ax)
    fig_1[i] = fig
    save("experiment_figures/StackedPlot_0_$i.pdf", fig, px_per_unit=2)
    save("experiment_figures/StackedPlot_0_$i.eps", fig, px_per_unit=2)
    save("experiment_figures/StackedPlot_0_$i.png", fig, px_per_unit=2)
end
fig_2 = Vector{Figure}(undef, 3)
for i in 1:3
    fig = Figure(fontsize=fontsize)
    ax = Axis(fig[1, 1])
    scatter!(ax, data_set_x_2[i], data_set_y_2[i], color=colors[i], markersize=4)
    hidedecorations!(ax)
    fig_2[i] = fig
    save("experiment_figures/StackedPlot_48_$i.pdf", fig, px_per_unit=2)
    save("experiment_figures/StackedPlot_48_$i.eps", fig, px_per_unit=2)
    save("experiment_figures/StackedPlot_48_$i.png", fig, px_per_unit=2)
end

## Fourth figure: Measuring cell density
fig = Figure(fontsize=fontsize)
ax = Axis(fig[1, 1], title="(c): Measuring cell density", titlealign=:left)
scatter!(ax, measuring_x_1[1], measuring_y_1[1], markersize=4, color=:black)
vlines!(ax, 0:50:1900, color=:blue)
ax.xticks = 0:(19*25):1900
hideydecorations!(ax)
hidexdecorations!(ax, label=false, ticklabels=false)
save("experiment_figures/MeasuringPlotsu.pdf", fig, px_per_unit=2)
save("experiment_figures/MeasuringPlotsu.eps", fig, px_per_unit=2)
save("experiment_figures/MeasuringPlotsu.png", fig, px_per_unit=2)

## Fifth figure: Measuring cell carrying capacity density 
fig = Figure(fontsize=fontsize)
ax = Axis(fig[1, 1], title="(d): Measuring cell carrying capacity", titlealign=:left)
scatter!(ax, measuring_x_K[1], measuring_y_K[1], markersize=4, color=:black)
K_window = 200
ax.xticks = 0:(19*25):1900
vlines!(ax, [50, K_window + 50, 1650, K_window + 1650], color=:blue)
hideydecorations!(ax)
hidexdecorations!(ax, label=false, ticklabels=false)
save("experiment_figures/MeasuringPlotsK.pdf", fig, px_per_unit=2)
save("experiment_figures/MeasuringPlotsK.eps", fig, px_per_unit=2)
save("experiment_figures/MeasuringPlotsK.png", fig, px_per_unit=2)
