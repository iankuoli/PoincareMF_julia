#
# For Poincare MF
#
vec_sq_norm_theta = sum(matTheta[:, usr_idx].^2, 1)[:]
vec_a = 1 .- vec_sq_norm_theta
vec_sq_norm_beta = sum(matBeta[:, itm_idx].^2, 1)[:]
vec_b = 1 .- vec_sq_norm_beta

usr_idx = collect(1:40)
itm_idx = collect(1:30)

vec_sq_norm_theta = sum(matTheta[:, usr_idx].^2, 1)[:]
vec_a = 1 .- vec_sq_norm_theta
vec_sq_norm_beta = sum(matBeta[:, itm_idx].^2, 1)[:]
vec_b = 1 .- vec_sq_norm_beta

mmm = zeros(40,30)
usr_idx_len=40

for u = 1:usr_idx_len
  u_id = usr_idx[u]
  tmp_theta = matTheta[:,u_id]
  sq_l2_dist_ui = sum(broadcast(+, matBeta[:,itm_idx], -tmp_theta).^2, 1)[:]
  val_d_u = acosh.( 1 .+ 2 * vec_sq_norm_theta[u] ./ vec_a[u] )
  vec_d_i = acosh.( 1 .+ 2 * vec_sq_norm_beta ./ vec_b )
  vec_d_ui = acosh.( 1 .+ 2 * sq_l2_dist_ui ./ vec_a[u] ./ vec_b )
  vec_s_ui = 0.5 * ( val_d_u.^2 .+ vec_d_i.^2 .- vec_d_ui.^2 )
  vec_f_ui = 1 ./ ( 1 .+ exp.(-vec_s_ui) )
  mmm[u,:] = alpha * matX_train[u_id, itm_idx] .* log.(vec_f_ui) + log.(1. .- vec_f_ui)
end

Plots.surface(mmm)



#
# For LogisticMF
#
usr_idx_len=40
mmm = zeros(40,30)

for u = 1:usr_idx_len
  u_id = usr_idx[u]
  vec_f_ui = 1 ./ ( 1 .+ exp.(-matBeta * matTheta[u_id,:]) )
  mmm[u,:] = alpha * matX_train[u_id, itm_idx] .* log.(vec_f_ui) + log.(1. .- vec_f_ui)
end

Plots.surface(mmm)
