module PoseEstimation

# https://github.com/dev10110/GraduatedNonConvexity.jl
using LinearAlgebra, StaticArrays, GraduatedNonConvexity, Parameters
# https://github.com/dev10110/ParallelMaximumClique.jl
using Graphs
using ParallelMaximumClique # For max clique. See also SimpleWeightedGraph, but not compatible with ParallelMaximumClique
using Random, Rotations
using Printf, Logging
debug_logger = Logging.ConsoleLogger(Logging.Info)

SV3{F} = SVector{3,F}
SV4{F} = SVector{4,F}
SM3{F} = SMatrix{3,3,F,9}
Quaternion{F} = SVector{4,F}  # x y z w

function rotdist(R1, R2)
    s = (tr(R1 * R2') - 1) / 2
    s = clamp(s, -1.0, 1.0) # to correct for potential float pt errors
    return acos(s)
end

# returns qa ∘ qb
function quatprod(qa, qb)
    return Ω1(qa) * qb
end

# returns inv(q)
function quatinv(q::Quaternion)
    return Quaternion(-q[1], -q[2], -q[3], q[4])
end

# returns operator Ω1(q) such that q ∘ v = Ω1(q) v̂
function Ω1(q::Quaternion)# left mul operator

    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    q4 = q[4]

    return @SMatrix [
        [q4;; -q3;; q2;; q1];
        [q3;; q4;; -q1;; q2];
        [-q2;; q1;; q4;; q3];
        [-q1;; -q2;; -q3;; q4]
    ]

end
function Ω1(q::SV3)
    return Ω1(Quaternion(q..., 0))
end

# returns operator Ω2(q) such that v ∘ q = Ω2(q) v̂
function Ω2(q::Quaternion) # right mul operator

    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    q4 = q[4]

    return @SMatrix [
        [q4;; q3;; -q2;; q1];
        [-q3;; q4;; q1;; q2];
        [q2;; -q1;; q4;; q3];
        [-q1;; -q2;; -q3;; q4]
    ]

end

function Ω2(q::SV3)
    return Ω2(Quaternion(q..., 0))
end

function quat_to_rot(q::Quaternion{F}) where {F}
    R̂ = Ω1(q) * Ω2(quatinv(q))
    R = R̂[1:3, 1:3]
    return SMatrix{3,3,F,9}(R)
end

function construct_Q(N, a::AF, b::AF, w) where {F,AF<:AbstractArray{F}}
    # construct Q matrix
    Q = zero(MMatrix{4,4,F,16})

    @inbounds for i = 1:N
        if w[i] != 0
            qa = Quaternion(a[1, i], a[2, i], a[3, i], zero(F))
            qb = Quaternion(b[1, i], b[2, i], b[3, i], zero(F))
            Ω1b = Ω1(qb)
            Ω2a = Ω2(qa)
            Qi = Ω1b' * Ω2a
            wQiQi = w[i] * (Qi + Qi')
            Q .-= wQiQi
        end
    end
    return Q
end

function construct_Q_fast(N, a::AF, b::AF, w) where {F,AF<:AbstractArray{F}}

    # allocate the Q matrix
    # note, this is a symmetric matrix!
    Q = zero(MMatrix{4,4,F,16})

    @inbounds for i = 1:N
        qa1 = a[1, i]
        qa2 = a[2, i]
        qa3 = a[3, i]
        qb1 = b[1, i]
        qb2 = b[2, i]
        qb3 = b[3, i]
        wi = w[i]

        Q[1, 1] -= 2*(qa1*qb1 - qa2*qb2 - qa3*qb3)*wi
        Q[1, 2] -= 2*(qa2*qb1 + qa1*qb2)*wi
        Q[1, 3] -= 2*(qa3*qb1 + qa1*qb3)*wi
        Q[1, 4] -= (-2*qa3*qb2 + 2*qa2*qb3)*wi
        Q[2, 1] -= 2*(qa2*qb1 + qa1*qb2)*wi
        Q[2, 2] -= -2*(qa1*qb1 - qa2*qb2 + qa3*qb3)*wi
        Q[2, 3] -= 2*(qa3*qb2 + qa2*qb3)*wi
        Q[2, 4] -= 2*(qa3*qb1 - qa1*qb3)*wi
        Q[3, 1] -= 2*(qa3*qb1 + qa1*qb3)*wi
        Q[3, 2] -= 2*(qa3*qb2 + qa2*qb3)*wi
        Q[3, 3] -= -2*(qa1*qb1 + qa2*qb2 - qa3*qb3)*wi
        Q[3, 4] -= (-2*qa2*qb1 + 2*qa1*qb2)*wi
        Q[4, 1] -= (-2*qa3*qb2 + 2*qa2*qb3)*wi
        Q[4, 2] -= 2*(qa3*qb1 - qa1*qb3)*wi
        Q[4, 3] -= (-2*qa2*qb1 + 2*qa1*qb2)*wi
        Q[4, 4] -= 2*(qa1*qb1 + qa2*qb2 + qa3*qb3)*wi

    end
    return Q
end

# lastest
function construct_full_Q_matrix(a::AF, b::AF) where {F,AF<:AbstractArray{F}}

    N = size(a, 2)
    @assert N == size(b, 2)

    # allocate the Q matrix
    Q = zeros(F, N, 16)

    L = LinearIndices((4, 4))

    @inbounds for i = 1:N
        qa1 = a[1, i]
        qa2 = a[2, i]
        qa3 = a[3, i]
        qb1 = b[1, i]
        qb2 = b[2, i]
        qb3 = b[3, i]

        Q[i, L[1, 1]] -= 2*(qa1*qb1 - qa2*qb2 - qa3*qb3)
        Q[i, L[1, 2]] -= 2*(qa2*qb1 + qa1*qb2)
        Q[i, L[1, 3]] -= 2*(qa3*qb1 + qa1*qb3)
        Q[i, L[1, 4]] -= (-2*qa3*qb2 + 2*qa2*qb3)
        # Q[i, L[2,1]] -= 2*(qa2*qb1 + qa1*qb2)
        Q[i, L[2, 2]] -= -2*(qa1*qb1 - qa2*qb2 + qa3*qb3)
        Q[i, L[2, 3]] -= 2*(qa3*qb2 + qa2*qb3)
        Q[i, L[2, 4]] -= 2*(qa3*qb1 - qa1*qb3)
        # Q[i, L[3,1]] -= 2*(qa3*qb1 + qa1*qb3)
        # Q[i, L[3,2]] -= 2*(qa3*qb2 + qa2*qb3)
        Q[i, L[3, 3]] -= -2*(qa1*qb1 + qa2*qb2 - qa3*qb3)
        Q[i, L[3, 4]] -= (-2*qa2*qb1 + 2*qa1*qb2)
        # Q[i, L[4,1]] -= (-2*qa3*qb2 + 2*qa2*qb3)
        # Q[i, L[4,2]] -= 2*(qa3*qb1 - qa1*qb3)
        # Q[i, L[4,3]] -= (-2*qa2*qb1 + 2*qa1*qb2)
        Q[i, L[4, 4]] -= 2*(qa1*qb1 + qa2*qb2 + qa3*qb3)

    end
    return Q
end

"""
    estimate_R(a, b, w)

solves the problem

R^* = min sum_{i=1}^N w_i ||b_i - R a_i||^2

in closed form
"""
function _estimate_R!(
    R,
    a::AF,
    b::AF,
    w = ones(F, size(a, 2)),
) where {F,AF<:AbstractArray{F}}
    """
    In-place version of estimate_R.
    """
    @assert size(a) == size(b)
    N = size(a, 2)
    @assert length(w) == N
    Q = construct_Q(N, a, b, w)
    # get min eigenvector 
    q = eigvecs(Q)[:, 1] |> real |> Quaternion{F}
    # convert to rotation matrix
    R .= quat_to_rot(q)
end

"""
    estimate_t(s, w)

solves the problem

R^* = min sum_{i=1}^N w_i ||s_i - t||^2

in closed form

"""
function _estimate_t!(x, s, w = ones(size(s, 2)))
    D, N = size(s) # t \in R^D, and there are N points to match
    P = sum(w)*I(D)
    q = sum(w' .* s, dims = 2)
    ldiv!(x, factorize(P), q)
end

abstract type PairingMethod end

@with_kw struct Star <: PairingMethod
    n::Integer = 1
end

@with_kw struct Complete <: PairingMethod
    frac::Float64 = 1.0
end

function make_pairs(m::Star, N)
    is = ones(Int, N-1) * m.n
    js = [j for j = 1:N if j != m.n]
    return is, js
end

# TODO(rgg): add other sparse topologies?

function make_pairs(m::Complete, N::T) where {T}

    is = T[]
    js = T[]
    for i = 1:N, j = (i+1):N
        if rand() < m.frac
            push!(is, i)
            push!(js, j)
        end
    end

    return is, js
end

abstract type LsqMethod end

struct LS <: LsqMethod end

@with_kw struct GM <: LsqMethod
    c̄::Any
    max_iterations = 1000
    μ_factor = 1.4
    verbose = false
    rtol = 1e-6
end
GM(c, kwargs...) = GM(c̄ = c; kwargs...)

@with_kw struct TLS <: LsqMethod
    c̄::Any
    max_iterations = 1000
    μ_factor = 1.4
    rtol = 1e-6
    verbose = false
    # verbose = true
end
TLS(c) = TLS(c̄ = c)

function wls_solver_R!(x, w, data)
    a, b = data
    _estimate_R!(x, a, b, w)
end

# second version of wls_solver_R, Q is constructed else where
function wls_solver_R2!(x, w::VF, data) where {F,VF<:AbstractVector{F}}
    Qmat = data[3] # this is the uniform weights Q, vectorized
    wQ = Symmetric(reshape(w' * Qmat, 4, 4))

    q = eigvecs(wQ)[:, 1] |> Quaternion{F}

    x .= quat_to_rot(q)
end

# this is slow
function residuals_R!(rs, R, δ, data)
    a, b = data
    numerator = b - R*a
    for (i, col) in enumerate(eachcol(numerator))
        rs[i] = norm(col) / δ
    end
end

function residuals_R_fast!(rs, R, δ, data)
    # a, b = data
    a = data[1]
    b = data[2]
    vecnorm!(rs, b, R*a, 1/δ)
end

function residuals_R_fast_scaled!(rs, R, δ, data) # δ here is a vector for each entry
    # a, b = data
    a = data[1]
    b = data[2]
    vecnorm_scaled!(rs, b, R*a, δ)
end

# fast norm for substitution of Main.norm
function vecnorm!(
    r::VF,
    A::AVF,
    B::AF,
    c::F = 1,
) where {F<:AbstractFloat,VF<:AbstractVector{F},AF<:AbstractArray{F},AVF<:AbstractArray{F}}

    @inbounds for i = 1:size(A, 2)
        r[i] =
            c *
            sqrt((A[1, i] - B[1, i])^2 + (A[2, i] - B[2, i]) ^ 2 + (A[3, i] - B[3, i])^2)
    end
end

# fast norm for substitution of Main.norm
function vecnorm_scaled!(
    r::VF,
    A::AVF,
    B::AF,
    δ::VF,
) where {F<:AbstractFloat,VF<:AbstractVector{F},AF<:AbstractArray{F},AVF<:AbstractArray{F}}

    @inbounds for i = 1:size(A, 2)
        r[i] =
            (1/δ[i]) *
            sqrt((A[1, i] - B[1, i])^2 + (A[2, i] - B[2, i]) ^ 2 + (A[3, i] - B[3, i])^2)
    end
end

function wls_solver_t!(x, w, s)
    return _estimate_t!(x, s, w)
end

# this is slow
function residuals_t!(rs, t, s, β)
    rs = norm.(eachcol(s .- t)) / β
end

function residuals_t_fast!(rs, t, s, β)
    vecnorm!(rs, t, s, 1/β)
end

function residuals_t_fast_scaled!(rs, t, s, δvec)
    vecnorm_scaled!(rs, t, s, δvec)
end

# function estimate_R_fast(method::TLS, a::Matrix{V}, b::Matrix{V}, δ) where {V}
#     """
#     In-place version of estimate_R.
#     Only uses TLS (no multiple dispatch) for reduced precompilation time.
#     """
#     N = size(a, 2)
#     data = (a, b)
#     w = ones(V, N)
#     rs = Vector{V}(undef, N)
#     R = collect(I(3)*1f0)
#     # Required initialization.
#     # Without this, GNC_TLS may converge to a bad solution / throw errors
#     wls_solver_R!(R, w, data)  
#     GNC_TLS!(R, w, rs, data, wls_solver_R!, (rs, R, data)->residuals_R!(rs, R, δ, data), method.c̄;
#         max_iterations = method.max_iterations,
#         μ_factor = method.μ_factor,
#         verbose=method.verbose,
#         rtol = method.rtol
#     )
#     return R
# end

function estimate_R_fast_fast(method::TLS, a::Matrix{V}, b::Matrix{V}, δ) where {V}
    """
    In-place version of estimate_R.
    """
    N = size(a, 2)
    Qmat = construct_full_Q_matrix(a, b) # first compute Q
    data = (a, b, Qmat)
    w = ones(V, N)
    rs = Vector{V}(undef, N)
    R = collect(I(3)*1.0f0)
    wls_solver_R2!(R, w, data)
    GNC_TLS!(
        R,
        w,
        rs,
        data,
        wls_solver_R2!,
        (rs, R, data)->residuals_R_fast!(rs, R, δ, data),
        method.c̄;
        max_iterations = method.max_iterations,
        μ_factor = method.μ_factor,
        verbose = method.verbose,
        rtol = method.rtol,
    )
    return R
end

function estimate_R_fast_fast_scaled(
    method::TLS,
    a::Matrix{V},
    b::Matrix{V},
    δvec,
) where {V}

    """
    In place version of estimate R, but considers the δ as a vector
    """
    N = size(a, 2)
    Qmat = construct_full_Q_matrix(a, b) # first compute Q
    data = (a, b, Qmat)
    w = ones(V, N)
    rs = Vector{V}(undef, N)
    R = collect(I(3)*1.0f0)
    wls_solver_R2!(R, w, data)
    GNC_TLS!(
        R,
        w,
        rs,
        data,
        wls_solver_R2!,
        (rs, R, data)->residuals_R_fast_scaled!(rs, R, δvec, data),
        method.c̄;
        max_iterations = method.max_iterations,
        μ_factor = method.μ_factor,
        verbose = method.verbose,
        rtol = method.rtol,
    )
    return R

end

function estimate_t_fast(method::TLS, p1::Matrix{V}, p2::Matrix{V}, R, β) where {V}
    """
    In-place version of estimate_t.
    Only uses TLS (no multiple dispatch) for reduced precompilation time.
    """
    # Method provided for options data, not for type inference
    s = p2 - R * p1
    N = size(s, 2)
    w = ones(V, N)
    # Initialize rs to vector of all 1s
    rs = ones(V, N)
    t = zeros(Float32, 3)
    # Required initialization.
    # Without this, GNC_TLS may converge to a bad solution / throw errors
    wls_solver_t!(t, w, s)
    GNC_TLS!(
        t,
        w,
        rs,
        s,
        wls_solver_t!,
        (rs, t, s)->residuals_t_fast!(rs, t, s, β),
        method.c̄;
        max_iterations = method.max_iterations,
        μ_factor = method.μ_factor,
        verbose = method.verbose,
        rtol = method.rtol,
    )
    return t
end

function estimate_t_fast_scaled(
    method::TLS,
    p1::Matrix{V},
    p2::Matrix{V},
    R,
    δvec,
) where {V}
    """
    In-place version of estimate_t.
    Only uses TLS (no multiple dispatch) for reduced precompilation time.
    Also uses the scaled δ errors
    """
    # Method provided for options data, not for type inference
    s = p2 - R * p1
    N = size(s, 2)
    w = ones(V, N)
    # Initialize rs to vector of all 1s
    rs = ones(V, N)
    t = zeros(Float32, 3)
    # Required initialization.
    # Without this, GNC_TLS may converge to a bad solution / throw errors
    wls_solver_t!(t, w, s)
    GNC_TLS!(
        t,
        w,
        rs,
        s,
        wls_solver_t!,
        (rs, t, s)->residuals_t_fast_scaled!(rs, t, s, δvec),
        method.c̄;
        max_iterations = method.max_iterations,
        μ_factor = method.μ_factor,
        verbose = method.verbose,
        rtol = method.rtol,
    )
    return t
end

function estimate_Rt_fast(
    p1::Matrix,
    p2::Matrix;
    β::Float32,
    method_pairing::PairingMethod,
    method_R::TLS,
    method_t::TLS,
)
    """
    Uses in-place operations to speed up TLS methods.
    """
    N = size(p1, 2)
    # In order to estimate rototranslation, we need Translation Invariant Measurements (TIMs) 
    # across the two frames. This allows for outlier rejection.
    # Pairing method is configurable and not directly related to the keypoints themselves.
    is, js = make_pairs(method_pairing, N)
    a = p1[:, is] - p1[:, js]
    b = p2[:, is] - p2[:, js]
    R = estimate_R_fast_fast(method_R, a, b, 2*β)
    t = estimate_t_fast(method_t, p1, p2, R, β)
    return R, t
end

function estimate_Rt_fast_scaled(
    p1::Matrix,
    p2::Matrix;
    δ_scale_law,
    method_pairing::PairingMethod,
    method_R::TLS,
    method_t::TLS,
)
    """
    Uses in-place operations to speed up TLS methods.
    Also uses the scaled versions of error bounds
    δ_scale_law is a fucntion that maps norm to error distance
    """
    N = size(p1, 2)
    # In order to estimate rototranslation, we need Translation Invariant Measurements (TIMs) 
    # across the two frames. This allows for outlier rejection.
    # Pairing method is configurable and not directly related to the keypoints themselves.
    is, js = make_pairs(method_pairing, N)
    a = p1[:, is] - p1[:, js]
    b = p2[:, is] - p2[:, js]
    # create the δij vector
    δij = Vector{Float32}(undef, length(is))
    for k = 1:length(is)
        nai = norm(p1[:, is[k]])
        naj = norm(p1[:, js[k]])
        δij[k] = δ_scale_law(nai) + δ_scale_law(naj)
    end
    δi = Vector{Float32}(undef, N)
    for i = 1:N
        nai = norm(p1[:, i])
        δi[i] = δ_scale_law(nai);
    end

    R = estimate_R_fast_fast_scaled(method_R, a, b, δij)
    t = estimate_t_fast_scaled(method_t, p1, p2, R, δi)
    return R, t
end

function get_inlier_inds(p1::Matrix, p2::Matrix, ϵ, method_pairing::PairingMethod)
    """
    Use translation invariant measurements to find inlier indices using max-clique inlier selection.
    Returns: list of indices 
    Args:
        p1: 3xN list of points in frame 1
        p2: 3xN list of points in frame 2 that correspond column-wise to points in p1.
            May contain false correspondences, and noise determined by sensing & feature detection.
        ϵ: maximum noise for inlier correspondences
        method_pairing: pairing method to create TIMs from keypoints
    Assumes scaling is unity. TODO(rgg): implement scale estimation?
    Optimal (most accurate) pairing method is to form a complete graph.
    """
    N = size(p1, 2)
    print("N: ", N)
    # Create TIMs (see: TEASER paper)
    is, js = make_pairs(method_pairing, N)  # TODO(rgg): use Graphs.jl throughout?
    E = length(is)
    # @info @sprintf("Number of edges in TIM graph: %i\n", E)
    # Vectors from keypoints in frame 1 to other keypoints in frame 1
    # Columns in ̄a correspond to columns in ̄b
    tim_̄a = p1[:, is] - p1[:, js]
    tim_̄b = p2[:, is] - p2[:, js]
    println("num of edges: ", size(tim_̄a))
    # Create TRIMs
    G = SimpleGraph(N)  # Unweighted edges for compatibility with max clique library
    # Skip edges that are not consistent with estimated scale (assume s=1 for now)
    # s_ij = s + o_ij^s + ϵ_ij^s; TRIM is equal to true scaling + outlier noise + modeled noise 
    # relative noise |ϵ_ij| ≤ 2*|ϵ| as the ϵ is the noise for measurements in each frame
    ϵ_ij = 2*ϵ
    println("relative noise: ", ϵ_ij)
    ϵ_ij_s = ϵ_ij ./ norm.(eachcol(tim_̄a))
    println("noise: ", size(ϵ_ij_s))
    s_expected = 1  # Assumption (static environment, camera parameters)
    # Compute scale estimate for each TRIM (edge weights in graph)
    s = norm.(eachcol(tim_̄b)) ./ norm.(eachcol(tim_̄a))
    println("scale: ", size(s))
    # Construct graph from TRIMs
    for k = 1:E
        # All TRIMs with scale outside of s ± ϵ_ij_s are inconsistent, do not add these edges
        @debug @sprintf(
            "Processing edge %i-%i with s=%f and scale bounds +/-%f\n",
            is[k],
            js[k],
            s[k],
            ϵ_ij_s[k]
        )
        if s_expected-ϵ_ij_s[k] <= s[k] <= s_expected+ϵ_ij_s[k]
            add_edge!(G, is[k], js[k])
        end
    end
    # @info @sprintf("Number of edges in pruned TRIM graph: %i\n", length(edges(G)))

    # Find maximum clique in remaining graph to get inliners
    # and return indices of vertices (points) that belong to inlier TRIMs.
    # @show G
    return maximum_clique(G)
end

σmin(A) = sqrt(max(0, eigmin(A'*A)))

# note, this function scales as N^4
# only pass in inliers to p1, i.e. pass in p1[:, inliner_idx]
# this function implements different maths than in Teaser
# since the proof in Teaser is wrong
# a good anytime approximation will be to run this function with random sets of 4 inds
# and take the lowest bound produced. This approximation will still be an upper-bound to the error
function ϵR(p1, β)
    N = size(p1, 2)
    best_bound = Inf
    @views @inbounds for i = 1:N
        PI = SMatrix{3,3}(p1[:, [i, i, i]])
        for j = (i+1):N, h = (j+1):N, k = (h+1):N
            U = SMatrix{3,3}(p1[:, [j, h, k]]) - PI
            σ = σmin(U)
            if σ > 0
                bound = 2*sqrt(3)*(2β) / σ
                best_bound = min(bound, best_bound)
            end
        end
    end
    return best_bound
end

function ϵR2(p1, β)
    best_bound = Inf
    i, j, k, h = 1:4
    U = SMatrix{3,3}(p1[:, [j, h, k]]) - SMatrix{3,3}(p1[:, [i, i, i]])
    L = sqrt(
        1/(U[1, 1]^2 + U[2, 1]^2 + U[3, 1]^2) +
        1/(U[1, 2]^2 + U[2, 2]^2 + U[3, 2]^2) +
        1/(U[1, 3]^2 + U[2, 3]^2 + U[3, 3]^2),
    )
    σ = σmin(U)
    if σ > 0
        bound = 4*β / σ * L
        best_bound = min(bound, best_bound)
    end
    return best_bound
end

function ϵt(β)
    return (9 + 3*sqrt(3))*β
end

function est_err_bounds(p1, p2, β; iterations = 5000)
    """
    Estimate error bounds on R, t for the given set of correspondences.
    Uses scale consistency and max-clique to predict inliers.
    Bound gets tighter with more iterations, but is always conservative.
    Args:
        p1: 3xN points in frame 1
        p2: 3xN points in frame 2 that correspond to those in frame 1, column-wise
        β: bound on the noise for points/correspondences in each frame; |ϵ| ≤ β
    """
    N = size(p1, 2)
    inliers = get_inlier_inds(p1, p2, β, Complete(N))
    # @show inliers
    if length(inliers) < 20
        return 0.0f0, 0.0f0
    end
    best_ϵR2=Inf
    # Select random subsets of 4 inliers.
    for i = 1:iterations
        best_ϵR2 = min(best_ϵR2, ϵR2(p1[:, rand(inliers, 4)], β))
    end
    return best_ϵR2, ϵt(β)
end

end
