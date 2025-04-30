module estimateRT

include("PoseEstimation.jl")
using StaticArrays
using .PoseEstimation
PE = PoseEstimation

# function quat_from_rotmatrix(Rot) 
#     # xyzw
#     a2 = 1 + Rot[1,1] + Rot[2,2] + Rot[3,3]
#     if (a2 < eps())
#         a2 = 0.0000001
#     end
#     a = sqrt(a2)/2
#     b,c,d = (Rot[3,2]-Rot[2,3])/4a, (Rot[1,3]-Rot[3,1])/4a, (Rot[2,1]-Rot[1,2])/4a
#     return SVector{4, Float64}([b c d a])
# end

function estimate_RT_timed(p1, p2, T, N)
    @time estimate_RT(p1, p2, T, N)
    return
end

function estimate_RT(p1, p2, T, N)
    # β = 0.005f0 
    # β = 0.010f0 
    # define the function of how the erorr scales with distance
    δ_scale_law(d) = 0.005f0 * d# 2% error
    # δ_scale_law(d) = 0.01f0
    c̄ = 1.0f0
    # println("im here")
    # # @show  p1, p2
    matched_pts1 = reshape(p1, (3, :))
    matched_pts2 = reshape(p2, (3, :))

    # for ii = 1:size(matched_pts2, 2)
    #     # @show  matched_pts2[:, ii]
    # end

    R_tls_2_1, t_tls_2_1 = PE.estimate_Rt_fast_scaled(
        matched_pts2,
        matched_pts1;
        δ_scale_law = δ_scale_law,
        method_pairing = PE.Complete(frac = 0.1),
        method_R = PE.TLS(c̄), # TODO: fix c̄, put in the theoretically correct value based on β
        method_t = PE.TLS(c̄),
    ) # 2 to 1 changed to 1 to 2 for debug

    ϵR = 0.0
    ϵt = 0.0
    if N > 210
        # p2 = deepcopy(matched_pts2)
        # p1 = deepcopy(matched_pts1)
        # ϵR, ϵt = PE.est_err_bounds(p2, p1, β)
        # ϵR, ϵt = PE.est_err_bounds(matched_pts2, matched_pts1, β)
    end
    # ϵR, ϵt = PE.est_err_bounds(p2, p1, β)
    # max_dist = 5f0
    # norm_ball_err = ϵR * max_dist + ϵt
    # @show norm_ball_err

    T[1] = t_tls_2_1[1]
    T[2] = t_tls_2_1[2]
    T[3] = t_tls_2_1[3]
    T[4] = R_tls_2_1[1, 1]
    T[5] = R_tls_2_1[1, 2]
    T[6] = R_tls_2_1[1, 3]
    T[7] = R_tls_2_1[2, 1]
    T[8] = R_tls_2_1[2, 2]
    T[9] = R_tls_2_1[2, 3]
    T[10] = R_tls_2_1[3, 1]
    T[11] = R_tls_2_1[3, 2]
    T[12] = R_tls_2_1[3, 3]
    T[13] = ϵR
    T[14] = ϵt

    @show T
end

end
