#
# Mini-batch Stochastic Poincare MF
#
function update_matTheta_poincare(model_type::String, matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                  matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, CS::Float64, lambda::Float64, lr::Float64, probDropout::Float64,
                                  usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                                  train_itr::Int64, mu::Float64, mu2::Float64, mt::Array{Float64,2}, nt::Array{Float64,2}, mtB::Array{Float64,1}, ntB::Array{Float64,1})

  sampled_i, sampled_j, sampled_v = findnz(matX_train[usr_idx, itm_idx])

  vec_sq_norm_theta = sum(matTheta[:, usr_idx].^2, 1)[:]
  vec_a = 1 .- vec_sq_norm_theta
  vec_sq_norm_beta = sum(matBeta[:, itm_idx].^2, 1)[:]
  vec_b = 1 .- vec_sq_norm_beta

  K = size(matTheta,1)

  # Get the gradient for all users in [1:usr_idx_len]
  # and updatet matTheta & vecBiasU by Adam (Adaptive Moment Estimation)
  if model_type == "PoincareMF_minibatch"
    ret = @parallel hcat for u = 1:usr_idx_len
      learn_u(CS, alpha, matX_train[usr_idx[u], itm_idx],
               matTheta[:,usr_idx[u]], matBeta[:,itm_idx], vecBiasU[usr_idx[u]], vecBiasI[itm_idx],
               vec_sq_norm_theta[u], vec_sq_norm_beta, vec_a[u], vec_b,
               train_itr::Int64, lr, probDropout, lambda, mu, mu2, mt[:, usr_idx[u]], nt[:, usr_idx[u]], mtB[usr_idx[u]], ntB[usr_idx[u]])
    end
  elseif model_type == "sqdistPoincareMF_minibatch"
    gamma = 1.0
    ret = @parallel hcat for u = 1:usr_idx_len
      learn_u_sqdist(CS, alpha, matX_train[usr_idx[u], itm_idx],
                     matTheta[:,usr_idx[u]], matBeta[:,itm_idx], vecBiasU[usr_idx[u]], vecBiasI[itm_idx],
                     vec_sq_norm_theta[u], vec_sq_norm_beta, vec_a[u], vec_b,
                     train_itr::Int64, lr, probDropout, lambda, gamma, mu, mu2, mt[:, usr_idx[u]], nt[:, usr_idx[u]], mtB[usr_idx[u]], ntB[usr_idx[u]])
    end
  elseif model_type == "invPoincareMF_minibatch"
    ret = @parallel hcat for u = 1:usr_idx_len
      learn_u_inverse(CS, alpha, matX_train[usr_idx[u], itm_idx],
                      matTheta[:,usr_idx[u]], matBeta[:,itm_idx], vecBiasU[usr_idx[u]], vecBiasI[itm_idx],
                      vec_sq_norm_theta[u], vec_sq_norm_beta, vec_a[u], vec_b,
                      train_itr::Int64, lr, probDropout, lambda, mu, mu2, mt[:, usr_idx[u]], nt[:, usr_idx[u]], mtB[usr_idx[u]], ntB[usr_idx[u]])
    end
  elseif model_type == "PoincareMF_tfidf_minibatch"
    log_sum_X_i = log.(sum(spones(matX_train[:, itm_idx]), 1) .+ 1.)[:]
    ret = @parallel hcat for u = 1:usr_idx_len
      learn_u_tfidf(CS, alpha, matX_train[usr_idx[u], itm_idx], log_sum_X_i,
                    matTheta[:,usr_idx[u]], matBeta[:,itm_idx], vecBiasU[usr_idx[u]], vecBiasI[itm_idx],
                    vec_sq_norm_theta[u], vec_sq_norm_beta, vec_a[u], vec_b,
                    train_itr::Int64, lr, probDropout, lambda, mu, mu2, mt[:, usr_idx[u]], nt[:, usr_idx[u]], mtB[usr_idx[u]], ntB[usr_idx[u]])
    end
  else
end

  matTheta[:, usr_idx] = ret[1:K, :]
  mt[:, usr_idx] = ret[(K+1):2*K, :]
  nt[:, usr_idx] = ret[(2*K+1):3*K, :]
  vecBiasU[usr_idx] = ret[(3*K+1), :]
  mtB[usr_idx] = ret[(3*K+2), :]
  ntB[usr_idx] = ret[(3*K+3), :]

  return nothing
end

function update_matBeta_poincare(model_type::String, matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                 matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, CS::Float64, lambda::Float64, lr::Float64, probDropout::Float64,
                                 usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                                 itr::Int64, mu::Float64, mu2::Float64, mt::Array{Float64,2}, nt::Array{Float64,2}, mtB::Array{Float64,1}, ntB::Array{Float64,1})

  update_matTheta_poincare(model_type, matBeta, matTheta, vecBiasI, vecBiasU, matX_train', alpha, CS, lambda, lr, probDropout, itm_idx_len, usr_idx_len, itm_idx, usr_idx, itr, mu, mu2, mt, nt, mtB, ntB)
end


function obj_val(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                 vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                 matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, CS::Float64, lambda::Float64,
                 usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})
  ret = 0

  vec_sq_norm_theta = sum(matTheta[:, usr_idx].^2, 1)[:]
  vec_a = 1 .- vec_sq_norm_theta
  vec_sq_norm_beta = sum(matBeta[:, itm_idx].^2, 1)[:]
  vec_b = 1 .- vec_sq_norm_beta

  for u = 1:usr_idx_len
    u_id = usr_idx[u]
    tmp_theta = matTheta[:,u_id]
    sq_l2_dist_ui = sum(broadcast(+, matBeta[:,itm_idx], -tmp_theta).^2, 1)[:]
    val_d_u = acosh.( 1 .+ 2 * vec_sq_norm_theta[u] ./ vec_a[u] )
    vec_d_i = acosh.( 1 .+ 2 * vec_sq_norm_beta ./ vec_b )
    vec_d_ui = acosh.( 1 .+ 2 * sq_l2_dist_ui ./ vec_a[u] ./ vec_b )
    vec_s_ui = CS * ( val_d_u.^2 .+ vec_d_i.^2 .- vec_d_ui.^2 + vecBiasU[u_id] + vecBiasI[itm_idx])
    vec_f_ui = 1 ./ ( 1 .+ exp.(-vec_s_ui) )
    ret += sum(alpha * matX_train[u_id, itm_idx] .* log.(vec_f_ui .+ 10e-20) + log.(1. .- vec_f_ui .+ 10e-20))
  end

  regularizer = 0.5 * lambda * (sum(matTheta.^2) + sum(matBeta.^2))
  ret -= regularizer

  return ret / usr_idx_len / itm_idx_len
end


function obj_val_sqdist(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                        vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                        matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, CS::Float64, lambda::Float64, gamma::Float64,
                        usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})
  ret = 0

  vec_sq_norm_theta = sum(matTheta[:, usr_idx].^2, 1)[:]
  vec_a = 1 .- vec_sq_norm_theta
  vec_sq_norm_beta = sum(matBeta[:, itm_idx].^2, 1)[:]
  vec_b = 1 .- vec_sq_norm_beta
  vec_d_u = acosh.( 1 .+ 2 * vec_sq_norm_theta ./ vec_a )

  for u = 1:usr_idx_len
    u_id = usr_idx[u]
    tmp_theta = matTheta[:,u_id]
    sq_l2_dist_ui = sum(broadcast(+, matBeta[:,itm_idx], -tmp_theta).^2, 1)[:]
    vec_d_ui = acosh.( 1 .+ 2 * sq_l2_dist_ui ./ vec_a[u] ./ vec_b )
    val_center = 1 ./ ( 1. + exp( vec_d_u[u]^2 - vecBiasU[u_id]) )
    vec_s_ui = -CS * vec_d_ui.^2 + vecBiasU[u_id] + vecBiasI[itm_idx]
    vec_f_ui = 1 ./ ( 1 .+ exp.(-vec_s_ui) )
    ret += sum(alpha * matX_train[u_id, itm_idx] .* log.(vec_f_ui .+ 10e-20) + log.(1. .- vec_f_ui .+ 10e-20)) + log(val_center + 10e-20) + log(1 - val_center + 10e-20)
  end

  regularizer = 0.5 * lambda * (sum(matTheta.^2) + sum(matBeta.^2))
  ret -= regularizer

  return ret / usr_idx_len / itm_idx_len
end


function obj_val_invrese(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                         vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                         matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, CS::Float64, lambda::Float64,
                         usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})
  ret = 0

  vec_sq_norm_theta = sum(matTheta[:, usr_idx].^2, 1)[:]
  vec_a = 1 .- vec_sq_norm_theta
  vec_sq_norm_beta = sum(matBeta[:, itm_idx].^2, 1)[:]
  vec_b = 1 .- vec_sq_norm_beta

  for u = 1:usr_idx_len
    u_id = usr_idx[u]
    tmp_theta = matTheta[:,u_id]
    sq_l2_dist_ui = sum(broadcast(+, matBeta[:,itm_idx], -tmp_theta).^2, 1)[:]
    val_d_u = acosh.( 1 .+ 2 * vec_sq_norm_theta[u] ./ vec_a[u] )
    vec_d_i = acosh.( 1 .+ 2 * vec_sq_norm_beta ./ vec_b )
    vec_d_ui = acosh.( 1 .+ 2 * sq_l2_dist_ui ./ vec_a[u] ./ vec_b )
    vec_s_ui = CS * ( (1. .- val_d_u).^2 .+ (1. .- vec_d_i).^2 .- vec_d_ui.^2 + vecBiasU[u_id] + vecBiasI[itm_idx])
    vec_f_ui = 1 ./ ( 1 .+ exp.(-vec_s_ui) )
    ret += sum(alpha * matX_train[u_id, itm_idx] .* log.(vec_f_ui .+ 10e-20) + log.(1. .- vec_f_ui .+ 10e-20))
  end

  regularizer = 0.5 * lambda * (sum(matTheta.^2) + sum(matBeta.^2))
  ret -= regularizer

  return ret / usr_idx_len / itm_idx_len
end


function obj_val_tfidf(matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                       matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, CS::Float64, lambda::Float64,
                       usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})
  ret = 0

  vec_sq_norm_theta = sum(matTheta[:, usr_idx].^2, 1)[:]
  vec_a = 1 .- vec_sq_norm_theta
  vec_sq_norm_beta = sum(matBeta[:, itm_idx].^2, 1)[:]
  vec_b = 1 .- vec_sq_norm_beta

  log_sum_X_i = log.(sum(spones(matX_train[:, itm_idx]), 1))[:]

  for u = 1:usr_idx_len
    u_id = usr_idx[u]
    tmp_theta = matTheta[:,u_id]
    sq_l2_dist_ui = sum(broadcast(+, matBeta[:,itm_idx], -tmp_theta).^2, 1)[:]
    val_d_u = acosh.( 1 .+ 2 * vec_sq_norm_theta[u] ./ vec_a[u] )
    vec_d_i = acosh.( 1 .+ 2 * vec_sq_norm_beta ./ vec_b )
    vec_d_ui = acosh.( 1 .+ 2 * sq_l2_dist_ui ./ vec_a[u] ./ vec_b )
    vec_s_ui = CS * ( val_d_u.^2 .+ vec_d_i.^2 .- vec_d_ui.^2 + vecBiasU[u_id] + vecBiasI[itm_idx])
    vec_f_ui = 1 ./ ( 1 .+ exp.(-vec_s_ui) )
    ret += sum(alpha * (matX_train[u_id, itm_idx] ./ log_sum_X_i) .* log.(vec_f_ui .+ 10e-20) + log.(1. .- vec_f_ui .+ 10e-20))
  end

  regularizer = 0.5 * lambda * (sum(matTheta.^2) + sum(matBeta.^2))
  ret -= regularizer

  return ret / usr_idx_len / itm_idx_len
end


function learn_u(CS::Float64, alpha::Float64, matX_train_ui::SparseVector{Float64,Int64},
                 matTheta_u::Array{Float64,1}, tmp_beta::Array{Float64,2}, vecBiasU_uid::Float64, vecBiasI_iid::Array{Float64,1},
                 vec_sq_norm_theta_u::Float64, vec_sq_norm_beta::Array{Float64,1}, vec_a_u::Float64, vec_b::Array{Float64,1},
                 train_itr::Int64, lr::Float64, probDropout::Float64, lambda::Float64, mu::Float64, mu2::Float64,
                 mt_u::Array{Float64,1}, nt_u::Array{Float64,1}, mtB_u::Float64, ntB_u::Float64)
  # matTheta_u = matTheta[:,u_id]
  # tmp_beta = matBeta[:,itm_idx]
  # vec_sq_norm_theta_u = vec_sq_norm_theta[u]
  # vec_a_u = vec_a[u]
  # vecBiasU_uid = vecBiasU[u_id]
  # vecBiasI_iid = vecBiasI[itm_idx]
  # matX_train_ui = matX_train[u_id, itm_idx]
  # mt_u = mt[:, u_id]
  # nt_u = nt[:, u_id]
  # mtB_u = mtB[u_id]
  # ntB_u = ntB[u_id]

  dropout_mask = rand(length(matTheta_u)) .> probDropout

  # Compute vec_partial_L_by_d
  # size = (itm_idx_len)
  sq_l2_dist_ui = sum(broadcast(+, tmp_beta, -matTheta_u).^2, 1)[:]
  val_d_u = acosh.( 1 .+ 2 * vec_sq_norm_theta_u ./ vec_a_u )
  vec_d_i = acosh.( 1 .+ 2 * vec_sq_norm_beta ./ vec_b )
  vec_d_ui = acosh.( 1 .+ 2 * sq_l2_dist_ui ./ vec_a_u ./ vec_b )
  vec_s_ui = CS * ( val_d_u.^2 .+ vec_d_i.^2 .- vec_d_ui.^2 + vecBiasU_uid + vecBiasI_iid)
  vec_f_ui = 1 ./ ( 1. .+ exp.(-vec_s_ui) )
  vec_f_ui[isnan.(vec_f_ui)] = 1.
  vec_partial_L_by_d = alpha * matX_train_ui .* (1 - vec_f_ui) - vec_f_ui

  # Compute vec_partial_d_by_theta
  # size = (K, itm_idx_len)
  vec_c_ui = 1 .+ 2 / vec_a_u * sq_l2_dist_ui ./ vec_b
  tmp_ui = matTheta_u * ( (vec_sq_norm_beta' - 2 * (matTheta_u' * tmp_beta) .+ 1) / vec_a_u^2 ) .- tmp_beta / vec_a_u
  tmp_ui = broadcast(*, tmp_ui, 4 ./ (vec_b .* sqrt.(vec_c_ui.^2 .- 1))')

  vec_c_u =  1 .+ 2 / vec_a_u * vec_sq_norm_theta_u
  tmp_u = (matTheta_u ./ vec_a_u^2) * (4 ./ sqrt.(vec_c_u.^2 .- 1)')

  mat_partial_L_by_theta_u = 2 * CS * broadcast(+, -broadcast(*, tmp_ui, vec_d_ui'), tmp_u * val_d_u) * vec_partial_L_by_d
  vec_partial_L_by_biasU_u = CS * sum(vec_partial_L_by_d)

  # Update matTheta by Adam (Adaptive Moment Estimation)
  # Kingma, D., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
  gt = vec_a_u^2 / 4 * mat_partial_L_by_theta_u - lambda * matTheta_u
  mt_u = mu * mt_u + (1 - mu) * gt
  nt_u = mu2 * nt_u + (1 - mu2) * (gt .* gt)
  mt_u = mt_u ./ (1 - mu ^ train_itr)
  nt_u = nt_u ./ (1 - mu2 ^ train_itr)
  matTheta_u[dropout_mask] += lr * mt_u[dropout_mask] ./ (sqrt.(nt_u[dropout_mask]) .+ 10e-9)
  norm_tmp = norm(matTheta_u)
  if norm_tmp >= 1
    matTheta_u = matTheta_u / norm_tmp - 10e-6 * sign.(matTheta_u)
  end

  # Output to return
  gt = vec_partial_L_by_biasU_u - lambda * vecBiasU_uid
  mtB_u = mu * mtB_u + (1 - mu) * gt
  ntB_u = mu2 * ntB_u + (1 - mu2) * (gt .* gt)
  mtB_u = mtB_u ./ (1 - mu ^ train_itr)
  ntB_u = ntB_u ./ (1 - mu2 ^ train_itr)
  vecBiasU_uid += lr * mtB_u ./ (sqrt(ntB_u) + 10e-9)

  ret = vcat(matTheta_u, mt_u, nt_u, vecBiasU_uid, mtB_u, ntB_u)
  return ret

end


function learn_u_sqdist(CS::Float64, alpha::Float64, matX_train_ui::SparseVector{Float64,Int64},
                        matTheta_u::Array{Float64,1}, tmp_beta::Array{Float64,2}, vecBiasU_uid::Float64, vecBiasI_iid::Array{Float64,1},
                        vec_sq_norm_theta_u::Float64, vec_sq_norm_beta::Array{Float64,1}, vec_a_u::Float64, vec_b::Array{Float64,1},
                        train_itr::Int64, lr::Float64, probDropout::Float64, lambda::Float64, gamma::Float64, mu::Float64, mu2::Float64,
                        mt_u::Array{Float64,1}, nt_u::Array{Float64,1}, mtB_u::Float64, ntB_u::Float64)
  # matTheta_u = matTheta[:,u_id]
  # tmp_beta = matBeta[:,itm_idx]
  # vec_sq_norm_theta_u = vec_sq_norm_theta[u]
  # vec_a_u = vec_a[u]
  # vecBiasU_uid = vecBiasU[u_id]
  # vecBiasI_iid = vecBiasI[itm_idx]
  # matX_train_ui = matX_train[u_id, itm_idx]
  # mt_u = mt[:, u_id]
  # nt_u = nt[:, u_id]
  # mtB_u = mtB[u_id]
  # ntB_u = ntB[u_id]

  dropout_mask = rand(length(matTheta_u)) .> probDropout

  # Compute vec_partial_L_by_d
  # size = (itm_idx_len)
  sq_l2_dist_ui = sum(broadcast(+, tmp_beta, -matTheta_u).^2, 1)[:]
  vec_d_ui = acosh.( 1 .+ 2 * sq_l2_dist_ui ./ vec_a_u ./ vec_b )
  vec_s_ui = -CS * vec_d_ui.^2 + vecBiasU_uid + vecBiasI_iid
  vec_f_ui = 1 ./ ( 1. .+ exp.(-vec_s_ui) )
  vec_f_ui[isnan.(vec_f_ui)] = 1.
  vec_partial_L_by_d = ( alpha * matX_train_ui ./ vec_f_ui - 1 ./ (1 .- vec_f_ui) ) .* (1 - vec_f_ui) .* vec_f_ui
  val_d_u = acosh.( 1 .+ 2 * vec_sq_norm_theta_u ./ vec_a_u )

  # Compute vec_partial_d_by_theta
  # size = (K, itm_idx_len)
  vec_c_ui = 1 .+ 2 / vec_a_u * sq_l2_dist_ui ./ vec_b
  tmp_ui = matTheta_u * ( (vec_sq_norm_beta' - 2 * (matTheta_u' * tmp_beta) .+ 1) / vec_a_u^2 ) .- tmp_beta / vec_a_u
  tmp_ui = broadcast(*, tmp_ui, 4 ./ (vec_b .* sqrt.(vec_c_ui.^2 .- 1))')

  val_center = 1 ./ ( 1. + exp.( val_d_u^2 - vecBiasU_uid) )
  vec_c_u =  1 + 2 ./ vec_a_u * vec_sq_norm_theta_u
  tmp_u = (matTheta_u ./ vec_a_u^2) * (4 ./ sqrt.(vec_c_u.^2 - 1))

  mat_partial_L_by_theta_u = -2 * CS * broadcast(*, tmp_ui, vec_d_ui') * vec_partial_L_by_d - 2 * (gamma * (1-val_center) - val_center) * tmp_u
  vec_partial_L_by_biasU_u = sum(vec_partial_L_by_d)

  # Update matTheta by Adam (Adaptive Moment Estimation)
  # Kingma, D., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
  gt = vec_a_u^2 / 4 * mat_partial_L_by_theta_u - lambda * matTheta_u
  mt_u = mu * mt_u + (1 - mu) * gt
  nt_u = mu2 * nt_u + (1 - mu2) * (gt .* gt)
  mt_u = mt_u ./ (1 - mu ^ train_itr)
  nt_u = nt_u ./ (1 - mu2 ^ train_itr)
  matTheta_u[dropout_mask] += lr * mt_u[dropout_mask] ./ (sqrt.(nt_u[dropout_mask]) .+ 10e-9)
  norm_tmp = norm(matTheta_u)
  if norm_tmp >= 1
    matTheta_u = matTheta_u / norm_tmp - 10e-6 * sign.(matTheta_u)
  end

  # Output to return
  gt = vec_partial_L_by_biasU_u - lambda * vecBiasU_uid
  mtB_u = mu * mtB_u + (1 - mu) * gt
  ntB_u = mu2 * ntB_u + (1 - mu2) * (gt .* gt)
  mtB_u = mtB_u ./ (1 - mu ^ train_itr)
  ntB_u = ntB_u ./ (1 - mu2 ^ train_itr)
  vecBiasU_uid += lr * mtB_u ./ (sqrt(ntB_u) + 10e-9)
  #vecBiasU_uid < 0 ? vecBiasU_uid = 0 : vecBiasU_uid

  ret = vcat(matTheta_u, mt_u, nt_u, vecBiasU_uid, mtB_u, ntB_u)
  return ret

end


function learn_u_inverse(CS::Float64, alpha::Float64, matX_train_ui::SparseVector{Float64,Int64},
                         matTheta_u::Array{Float64,1}, tmp_beta::Array{Float64,2}, vecBiasU_uid::Float64, vecBiasI_iid::Array{Float64,1},
                         vec_sq_norm_theta_u::Float64, vec_sq_norm_beta::Array{Float64,1}, vec_a_u::Float64, vec_b::Array{Float64,1},
                         train_itr::Int64, lr::Float64, probDropout::Float64, lambda::Float64, mu::Float64, mu2::Float64,
                         mt_u::Array{Float64,1}, nt_u::Array{Float64,1}, mtB_u::Float64, ntB_u::Float64)
  # matTheta_u = matTheta[:,u_id]
  # tmp_beta = matBeta[:,itm_idx]
  # vec_sq_norm_theta_u = vec_sq_norm_theta[u]
  # vec_a_u = vec_a[u]
  # vecBiasU_uid = vecBiasU[u_id]
  # vecBiasI_iid = vecBiasI[itm_idx]
  # matX_train_ui = matX_train[u_id, itm_idx]
  # mt_u = mt[:, u_id]
  # nt_u = nt[:, u_id]
  # mtB_u = mtB[u_id]
  # ntB_u = ntB[u_id]

  dropout_mask = rand(length(matTheta_u)) .> probDropout

  # Compute vec_partial_L_by_d
  # size = (itm_idx_len)
  sq_l2_dist_ui = sum(broadcast(+, tmp_beta, -matTheta_u).^2, 1)[:]
  val_d_u = acosh.( 1 .+ 2 * vec_sq_norm_theta_u ./ vec_a_u )
  vec_d_i = acosh.( 1 .+ 2 * vec_sq_norm_beta ./ vec_b )
  vec_d_ui = acosh.( 1 .+ 2 * sq_l2_dist_ui ./ vec_a_u ./ vec_b )
  vec_s_ui = CS * ( (1 .- val_d_u).^2 .+ (1 .- vec_d_i).^2 .- vec_d_ui.^2 + vecBiasU_uid + vecBiasI_iid)
  vec_f_ui = 1 ./ ( 1. .+ exp.(-vec_s_ui) )
  vec_f_ui[isnan.(vec_f_ui)] = 1.
  vec_partial_L_by_d = alpha * matX_train_ui .* (1 - vec_f_ui) - vec_f_ui

  # Compute vec_partial_d_by_theta
  # size = (K, itm_idx_len)
  vec_c_ui = 1 .+ 2 / vec_a_u * sq_l2_dist_ui ./ vec_b
  tmp_ui = matTheta_u * ( (vec_sq_norm_beta' - 2 * (matTheta_u' * tmp_beta) .+ 1) / vec_a_u^2 ) .- tmp_beta / vec_a_u
  tmp_ui = broadcast(*, tmp_ui, 4 ./ (vec_b .* sqrt.(vec_c_ui.^2 .- 1))')

  vec_c_u =  1 .+ 2 / vec_a_u * vec_sq_norm_theta_u
  tmp_u = (matTheta_u ./ vec_a_u^2) * (4 ./ sqrt.(vec_c_u.^2 .- 1)')

  mat_partial_L_by_theta_u = 2 * CS * broadcast(-, -broadcast(*, tmp_ui, vec_d_ui'), tmp_u * val_d_u) * vec_partial_L_by_d
  vec_partial_L_by_biasU_u = CS * sum(vec_partial_L_by_d)

  # Update matTheta by Adam (Adaptive Moment Estimation)
  # Kingma, D., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
  gt = vec_a_u^2 / 4 * mat_partial_L_by_theta_u + lambda * matTheta_u
  mt_u = mu * mt_u + (1 - mu) * gt
  nt_u = mu2 * nt_u + (1 - mu2) * (gt .* gt)
  mt_u = mt_u ./ (1 - mu ^ train_itr)
  nt_u = nt_u ./ (1 - mu2 ^ train_itr)
  matTheta_u[dropout_mask] += lr * mt_u[dropout_mask] ./ (sqrt.(nt_u[dropout_mask]) .+ 10e-9)
  norm_tmp = norm(matTheta_u)
  if norm_tmp >= 1
    matTheta_u = matTheta_u / norm_tmp - 10e-6 * sign.(matTheta_u)
  end

  # Output to return
  gt = vec_partial_L_by_biasU_u - lambda * vecBiasU_uid
  mtB_u = mu * mtB_u + (1 - mu) * gt
  ntB_u = mu2 * ntB_u + (1 - mu2) * (gt .* gt)
  mtB_u = mtB_u ./ (1 - mu ^ train_itr)
  ntB_u = ntB_u ./ (1 - mu2 ^ train_itr)
  vecBiasU_uid += lr * mtB_u ./ (sqrt(ntB_u) + 10e-9)

  ret = vcat(matTheta_u, mt_u, nt_u, vecBiasU_uid, mtB_u, ntB_u)
  return ret

end


function learn_u_tfidf(CS::Float64, alpha::Float64, matX_train_ui::SparseVector{Float64,Int64}, log_sum_X_i::Array{Float64,1},
                       matTheta_u::Array{Float64,1}, tmp_beta::Array{Float64,2}, vecBiasU_uid::Float64, vecBiasI_iid::Array{Float64,1},
                       vec_sq_norm_theta_u::Float64, vec_sq_norm_beta::Array{Float64,1}, vec_a_u::Float64, vec_b::Array{Float64,1},
                       train_itr::Int64, lr::Float64, probDropout::Float64, lambda::Float64, mu::Float64, mu2::Float64,
                       mt_u::Array{Float64,1}, nt_u::Array{Float64,1}, mtB_u::Float64, ntB_u::Float64)
  # matTheta_u = matTheta[:,u_id]
  # tmp_beta = matBeta[:,itm_idx]
  # vec_sq_norm_theta_u = vec_sq_norm_theta[u]
  # vec_a_u = vec_a[u]
  # vecBiasU_uid = vecBiasU[u_id]
  # vecBiasI_iid = vecBiasI[itm_idx]
  # matX_train_ui = matX_train[u_id, itm_idx]
  # mt_u = mt[:, u_id]
  # nt_u = nt[:, u_id]
  # mtB_u = mtB[u_id]
  # ntB_u = ntB[u_id]

  dropout_mask = rand(length(matTheta_u)) .> probDropout

  # Compute vec_partial_L_by_d
  # size = (itm_idx_len)
  sq_l2_dist_ui = sum(broadcast(+, tmp_beta, -matTheta_u).^2, 1)[:]
  val_d_u = acosh.( 1 .+ 2 * vec_sq_norm_theta_u ./ vec_a_u )
  vec_d_i = acosh.( 1 .+ 2 * vec_sq_norm_beta ./ vec_b )
  vec_d_ui = acosh.( 1 .+ 2 * sq_l2_dist_ui ./ vec_a_u ./ vec_b )
  vec_s_ui = CS * ( val_d_u.^2 .+ vec_d_i.^2 .- vec_d_ui.^2 + vecBiasU_uid + vecBiasI_iid)
  vec_f_ui = 1 ./ ( 1. .+ exp.(-vec_s_ui) )
  vec_f_ui[isnan.(vec_f_ui)] = 1.
  vec_partial_L_by_d = alpha * (matX_train_ui ./ log_sum_X_i) .* (1 - vec_f_ui) - vec_f_ui

  # Compute vec_partial_d_by_theta
  # size = (K, itm_idx_len)
  vec_c_ui = 1 .+ 2 / vec_a_u * sq_l2_dist_ui ./ vec_b
  tmp_ui = matTheta_u * ( (vec_sq_norm_beta' - 2 * (matTheta_u' * tmp_beta) .+ 1) / vec_a_u^2 ) .- tmp_beta / vec_a_u
  tmp_ui = broadcast(*, tmp_ui, 4 ./ (vec_b .* sqrt.(vec_c_ui.^2 .- 1))')

  vec_c_u =  1 .+ 2 / vec_a_u * vec_sq_norm_theta_u
  tmp_u = (matTheta_u ./ vec_a_u^2) * (4 ./ sqrt.(vec_c_u.^2 .- 1)')

  mat_partial_L_by_theta_u = 2 * CS * broadcast(+, -broadcast(*, tmp_ui, vec_d_ui'), tmp_u * val_d_u) * vec_partial_L_by_d
  vec_partial_L_by_biasU_u = CS * sum(vec_partial_L_by_d)

  # Update matTheta by Adam (Adaptive Moment Estimation)
  # Kingma, D., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
  gt = vec_a_u^2 / 4 * mat_partial_L_by_theta_u - lambda * matTheta_u
  mt_u = mu * mt_u + (1 - mu) * gt
  nt_u = mu2 * nt_u + (1 - mu2) * (gt .* gt)
  mt_u = mt_u ./ (1 - mu ^ train_itr)
  nt_u = nt_u ./ (1 - mu2 ^ train_itr)
  matTheta_u[dropout_mask] += lr * mt_u[dropout_mask] ./ (sqrt.(nt_u[dropout_mask]) .+ 10e-9)
  norm_tmp = norm(matTheta_u)
  if norm_tmp >= 1
    matTheta_u = matTheta_u / norm_tmp - 10e-6 * sign.(matTheta_u)
  end

  gt = vec_partial_L_by_biasU_u - lambda * vecBiasU_uid
  mtB_u = mu * mtB_u + (1 - mu) * gt
  ntB_u = mu2 * ntB_u + (1 - mu2) * (gt .* gt)
  mtB_u = mtB_u ./ (1 - mu ^ train_itr)
  ntB_u = ntB_u ./ (1 - mu2 ^ train_itr)
  vecBiasU_uid += lr * mtB_u ./ (sqrt(ntB_u) + 10e-9)

  # Output to return
  ret = vcat(matTheta_u, mt_u, nt_u, vecBiasU_uid, mtB_u, ntB_u)
  return ret

end
