include("../src/PoseEstimation.jl")
using .PoseEstimation
using BenchmarkTools
using Random, Rotations
using LinearAlgebra
using Printf


PE = PoseEstimation

function generate_synthetic_data(;
    N = 1_000,
    β = 0.01,
    outlier_fraction = 0.05,
    outlier_noise = 0.1,
)
    """
    Generates synthetic point cloud data for testing pose estimation algorithms.
    Args:
        N: number of points to generate
        β: inlier noise bound
        outlier_fraction: fraction of points to make outliers
        outlier_noise: outlier noise multiplier. Noise will be gaussian with mean 0 and std = outlier_noise
    """
    # Generate a ground-truth pose
    R_groundtruth = rand(RotMatrix{3,Float32})
    # Generate a ground-truth translation
    t_groundtruth = randn(Float32, 3)
    # @show R_groundtruth, t_groundtruth
    # Generate points in frame 1
    p1 = randn(Float32, (3, N))
    # Generate true p2
    p2 = R_groundtruth * p1 .+ t_groundtruth
    # Make noisy measurements, bounded by inlier noise β
    p2_noisy = copy(p2)
    for i = 1:N
        ϵ = β * rand() * randn(Float32, 3) # random vector with norm <= β
        p2_noisy[:, i] += ϵ
    end

    # Add outliers to some percent of data. This noise exceeds inlier noise.
    outlier_inds = [i for i = 2:N if rand() < outlier_fraction]
    for i in outlier_inds
        p2_noisy[:, i] += outlier_noise*randn(3)
    end
    return p1, p2_noisy, R_groundtruth, t_groundtruth, outlier_inds
end

# Random.seed!(42); # # Set the random seed for reproducibility

β = 0.01f0
c̄ = 1
N = 200

data = generate_synthetic_data(N = N, β = β, outlier_fraction = 0.1, outlier_noise = 0.10)
p1, p2_noisy, R_groundtruth, t_groundtruth, outlier_inds = data

R_tls, t_tls = PE.estimate_Rt_fast(
    p1,
    p2_noisy;
    method_pairing = PE.Complete(frac = 0.1),
    β = β,
    method_R = PE.TLS(c̄ = c̄),
    method_t = PE.TLS(c̄ = c̄),
)

eR, et = PE.est_err_bounds(p1, p2_noisy, β)
eR_est = norm(R_tls - Matrix{Float32}(R_groundtruth))
et_est = norm(t_tls - t_groundtruth)

println("\n\n\n")
println("*************")
println("** RESULTS **")
println("*************")

println("True R:")
# @show R_groundtruth
show(stdout, "text/plain", R_groundtruth)
println()

println("Estimated R:")
show(stdout, "text/plain", R_tls)
println()
println()

println("True t:")
show(stdout, "text/plain", t_groundtruth)
println()

println("Estimated t:")
show(stdout, "text/plain", t_tls)
println()

println()


println("Rotation Errors:")
println("eR_est: ", round(eR_est, digits = 6), " [measured error]")
println("eR:     ", round(eR, digits = 6), " [theoretical error bound]")
if eR_est < eR
    println("eR_est < eR: true 🎉")
else
    println("eR_est < eR: false ❌")
end

println()

println("Translation Errors:")
println("et_est: ", round(et_est, digits = 6), " [measured error]")
println("et:     ", round(et, digits = 6), " [theoretical error bound]")
if et_est < et
    println("et_est < et: true 🎉")
else
    println("et_est < et: false ❌")
end


@assert eR_est < eR && et_est < et
