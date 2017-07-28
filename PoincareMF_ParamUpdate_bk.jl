function update_matTheta_poincare(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                                  vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                  matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64,
                                  usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})

  # Compute partial L w.r.t distance
  predict_X = broadcast(+, broadcast(+, matTheta[usr_idx,:] * matBeta[itm_idx,:]', vecBiasU[usr_idx]), vecBiasI[itm_idx]')
  matTmp = exp(predict_X)
  matTmp = matTmp ./ (1 + matTmp)
  matTmp[isnan(matTmp)] = 1
  matTmp = (1 + alpha * matX_train[usr_idx, itm_idx]) .* matTmp
  matTmp = alpha * matX_train[usr_idx, itm_idx] - matTmp

  # Update matTheta
  partial_theta = matTmp * matBeta[itm_idx, :] - 0.5 * lambda * matTheta[usr_idx,:]
  matTheta[usr_idx,:] += lr * partial_theta
  #grad_sqr_sum_theta += partial_theta .^ 2

  # Update vecBiasU
  partial_biasU = sum(matTmp, 2)[:]
  vecBiasU[usr_idx] += lr * partial_biasU
  #grad_sqr_sum_biasU += partial_biasU .^ 2

  return nothing
end


function update_matBeta(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                        vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                        matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64,
                        usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})

  update_matTheta(matBeta, matTheta, vecBiasI, vecBiasU, matX_train', alpha, lambda, itm_idx_len, usr_idx_len, itm_idx, usr_idx)
end


function update_matTheta_poincare2(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                                   vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                   matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64,
                                   usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})

  # vec_norm_beta = zeros(itm_idx_len)
  # vec_b = zeros(itm_idx_len)

  vec_sq_norm_beta = sum(matBeta[:, itm_idx].^2, 1)[:]
  vec_b = 1 .- vec_sq_norm_beta

  # for i = 1:itm_idx_len
  #   vec_norm_beta[i] = norm(matBeta[:, itm_idx[i]])
  # end
  # vec_b = 1 .- vec_norm_beta.^2

  @time @simd for u = 1:usr_idx_len
    u_id = usr_idx[u]

    # Compute partial L w.r.t distance
    # size = (itm_idx_len)
    norm_theta_u = norm(matTheta[:, u_id])
    a = 1 - norm_theta_u^2

    sq_l2_dist_ui = sum(broadcast(+, -matBeta[:,itm_idx], matTheta[:,u_id]).^2, 1)[:]
    vec_f_ui = exp.(acosh.( 1 .+ 2 * sq_l2_dist_ui ./ (a .* vec_b) ))
    #vec_partial_L_by_d = 1 ./ (-1 .+ vec_f_ui) .- (1 .+ alpha .* matX_train[u_id, itm_idx]) ./ (1 .+ vec_f_ui) .* vec_f_ui
    vec_partial_L_by_d = ( 1 ./ vec_f_ui .- (1 .+ alpha .* matX_train[u_id, itm_idx]) ./ (1 .+ vec_f_ui) ) .* vec_f_ui

    # Compute partial distance w.r.t theta_u
    # size = (K, itm_idx_len)
    vec_c = 1 .+ 2 / a * sq_l2_dist_ui ./ vec_b
    vec_partial_d_by_theta_u = matTheta[:,u_id] * ((vec_sq_norm_beta' - 2 * (matTheta[:,u_id]' * matBeta[:,itm_idx]) .+ 1) / a^2)
    vec_partial_d_by_theta_u .-= matBeta[:,itm_idx] / a
    vec_partial_d_by_theta_u = broadcast(*, vec_partial_d_by_theta_u, 4 ./ (vec_b .* sqrt(vec_c.^2 .- 1))')

    # Update matTheta
    partial_theta = (vec_partial_d_by_theta_u * vec_partial_L_by_d)[:]
    matTheta[:,u_id] += lr * a^2 / 4 * partial_theta
    norm_tmp = norm(matTheta[:,u_id])
    if norm_tmp >= 1
      matTheta[:,u_id] = matTheta[:,u_id] / norm_tmp - 10e-6 * sign.(matTheta[:,u_id])
    end
  end

  return nothing
end


function update_matBeta_poincare2(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                         vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                         matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64,
                         usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})

  update_matTheta_poincare2(matBeta, matTheta, vecBiasI, vecBiasU, matX_train', alpha, lambda, itm_idx_len, usr_idx_len, itm_idx, usr_idx)
end


#
# Logistic-based objective function
#
function update_matTheta_poincare3(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                                   vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                   matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64, lr::Float64,
                                   usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                                   train_itr::Int64, mu::Float64, mu2::Float64, mt::Array{Float64,2}, nt::Array{Float64,2})

  sampled_i, sampled_j, sampled_v = findnz(matX_train[usr_idx, itm_idx])

  vec_sq_norm_theta = sum(matTheta[:, usr_idx].^2, 1)[:]
  vec_a = 1 .- vec_sq_norm_theta
  vec_sq_norm_beta = sum(matBeta[:, itm_idx].^2, 1)[:]
  vec_b = 1 .- vec_sq_norm_beta

  mat_partial_L_by_theta = zeros(Float64, size(matTheta,1), usr_idx_len)
  vec_partial_L_by_gamma = zeros(Float64, usr_idx_len)

  for itr=1:length(sampled_i)
    u = sampled_i[itr]
    i = sampled_j[itr]
    u_id = usr_idx[u]
    i_id = itm_idx[i]

    # Compute vec_partial_L_by_d
    sample_size = min(11, itm_idx_len-nnz(matX_train[u_id,:])+1)

    # vec_sampled_neg_i_id = sample(1:itm_idx_len, min(10, itm_idx_len-nnz(matX_train[u_id,:]))+nnz(matX_train[u_id,:]), replace=false)
    # vec_sampled_neg_i_id = vec_sampled_neg_i_id[find(matX_train[u_id, itm_idx[vec_sampled_neg_i_id]] .== 0)[1:10]]
    # vec_sampled_neg_i_id = vcat(vec_sampled_neg_i_id, i)

    vec_sampled_neg_i_id = zeros(Int64, sample_size)
    neg_i_itr = 1
    while neg_i_itr < sample_size
      sampled_neg_i = rand(collect(1:itm_idx_len))
      sampled_neg_i_id = itm_idx[sampled_neg_i]
      if matX_train[u_id, sampled_neg_i_id] == 0 && length(find(vec_sampled_neg_i_id == sampled_neg_i)) == 0
        vec_sampled_neg_i_id[neg_i_itr] = sampled_neg_i
        neg_i_itr += 1
      end
    end
    vec_sampled_neg_i_id[neg_i_itr] = i

    tmp_theta = matTheta[:,u_id]
    tmp_beta = matBeta[:,itm_idx[vec_sampled_neg_i_id]]
    sq_l2_dist_ui = sum(broadcast(+, -tmp_beta, tmp_theta).^2, 1)[:]
    vec_f_ui = exp.(acosh.( 1 .+ 2 * sq_l2_dist_ui ./ (vec_a[u] .* vec_b[vec_sampled_neg_i_id]) ))
    vec_partial_L_by_d = ( 1. ./ (-1. .+ vec_f_ui) .- (1. .+ alpha .* matX_train[u_id, itm_idx[vec_sampled_neg_i_id]]) ./ (1. .+ vec_f_ui) ) .* vec_f_ui

    # Compute vec_partial_d_by_theta
    # size(K, sample_size + 1)
    vec_c = 1 .+ 2 / vec_a[u] * sq_l2_dist_ui ./ vec_b[vec_sampled_neg_i_id]
    vec_partial_d_by_theta_u = tmp_theta * ((vec_sq_norm_beta[vec_sampled_neg_i_id]' - 2 * (tmp_theta' * tmp_beta) .+ 1) / vec_a[u]^2) .- tmp_beta / vec_a[u]
    vec_partial_d_by_theta_u = broadcast(*, vec_partial_d_by_theta_u, 4 ./ (vec_b[vec_sampled_neg_i_id] .* sqrt.(vec_c.^2 .- 1))')

    mat_partial_L_by_theta[:, u] += vec_partial_d_by_theta_u * vec_partial_L_by_d
  end

  # Update matTheta by Adam (Adaptive Moment Estimation)
  # Kingma, D., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
  for u = 1:usr_idx_len
    u_id = usr_idx[u]
    gt = vec_a[u]^2 / 4 * mat_partial_L_by_theta[:,u] - lambda * matTheta[:,u_id]
    mt[:, u_id] = mu * mt[:, u_id] + (1 - mu) * gt
    nt[:, u_id] = mu2 * nt[:, u_id] + (1 - mu2) * (gt .* gt)
    mt[:, u_id] = mt[:, u_id] ./ (1 - mu ^ 600)
    nt[:, u_id] = nt[:, u_id] ./ (1 - mu2 ^ 600)
    matTheta[:,u_id] += lr * mt[:, u_id] ./ (sqrt.(nt[:, u_id]) .+ 10e-9)
    norm_tmp = norm(matTheta[:,u_id])
    if norm_tmp >= 1
      matTheta[:,u_id] = matTheta[:,u_id] / norm_tmp - 10e-6 * sign.(matTheta[:,u_id])
    end
  end

  return nothing
end

function update_matBeta_poincare3(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                                  vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                  matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64, lr::Float64,
                                  usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                                  itr::Int64, mu::Float64, mu2::Float64, mt::Array{Float64,2}, nt::Array{Float64,2})

  update_matTheta_poincare3(matBeta, matTheta, vecBiasI, vecBiasU, matX_train', alpha, lambda, lr, itm_idx_len, usr_idx_len, itm_idx, usr_idx, itr, mu, mu2, mt, nt)
end


#
# Softmax-based objective function
#
function update_matTheta_poincare4(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                                   vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                   matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64, lr::Float64,
                                   usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                                   itr::Int64, mu::Float64, mu2::Float64, mt::Array{Float64,2}, nt::Array{Float64,2})

  sampled_i, sampled_j, sampled_v = findnz(matX_train[usr_idx, itm_idx])

  vec_sq_norm_theta = sum(matTheta[:, usr_idx].^2, 1)[:]
  vec_a = 1 .- vec_sq_norm_theta
  vec_sq_norm_beta = sum(matBeta[:, itm_idx].^2, 1)[:]
  vec_b = 1 .- vec_sq_norm_beta

  mat_partial_L_by_theta = zeros(Float64, size(matTheta,1), usr_idx_len)
  vec_partial_L_by_gamma = zeros(Float64, usr_idx_len)

  for itr=1:length(sampled_i)
    u = sampled_i[itr]
    i = sampled_j[itr]
    u_id = usr_idx[u]
    i_id = itm_idx[i]

    # Compute vec_partial_L_by_d
    sample_size = min(11, itm_idx_len-nnz(matX_train[u_id,:])+1)

    vec_sampled_neg_i_id = zeros(Int64, sample_size)
    neg_i_itr = 1
    while neg_i_itr < sample_size
      sampled_neg_i = rand(collect(1:itm_idx_len))
      sampled_neg_i_id = itm_idx[sampled_neg_i]
      if matX_train[u_id, sampled_neg_i_id] == 0 && length(find(vec_sampled_neg_i_id == sampled_neg_i)) == 0
        vec_sampled_neg_i_id[neg_i_itr] = sampled_neg_i
        neg_i_itr += 1
      end
    end
    vec_sampled_neg_i_id[neg_i_itr] = i

    tmp_theta = matTheta[:,u_id]
    tmp_beta = matBeta[:,itm_idx[vec_sampled_neg_i_id]]
    sq_l2_dist_ui = sum(broadcast(+, -tmp_beta, tmp_theta).^2, 1)[:]
    vec_f_ui = exp.(-acosh.( 1 .+ 2 * sq_l2_dist_ui ./ (vec_a[u] .* vec_b[vec_sampled_neg_i_id]) ))

    # Compute vec_partial_d_by_theta
    # size(K, sample_size + 1)
    vec_c = 1 .+ 2 / vec_a[u] ./ vec_b[vec_sampled_neg_i_id] .* sq_l2_dist_ui
    vec_partial_d_by_theta_u = tmp_theta * ((vec_sq_norm_beta[vec_sampled_neg_i_id]' - 2 * (tmp_theta' * tmp_beta) .+ 1) / vec_a[u]^2) .- tmp_beta / vec_a[u]
    vec_partial_d_by_theta_u = broadcast(*, vec_partial_d_by_theta_u, 4 ./ (vec_b[vec_sampled_neg_i_id] .* sqrt.(vec_c.^2 .- 1))')

    mat_partial_L_by_theta[:, u] += alpha * matX_train[u_id, i_id] * (-vec_partial_d_by_theta_u[:,end] + (vec_partial_d_by_theta_u * vec_f_ui) / sum(vec_f_ui))
  end

  # Update matTheta by Adam (Adaptive Moment Estimation)
  # Kingma, D., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
  for u = 1:usr_idx_len
    u_id = usr_idx[u]
    gt = vec_a[u]^2 / 4 * mat_partial_L_by_theta[:,u] - lambda * matTheta[:,u_id]
    mt[:, u_id] = mu * mt[:, u_id] + (1 - mu) * gt
    nt[:, u_id] = mu2 * nt[:, u_id] + (1 - mu2) * (gt .* gt)
    mt[:, u_id] = mt[:, u_id] ./ (1 - mu ^ 600)
    nt[:, u_id] = nt[:, u_id] ./ (1 - mu2 ^ 600)
    matTheta[:,u_id] += lr * mt[:, u_id] ./ (sqrt.(nt[:, u_id]) .+ 10e-9)
    norm_tmp = norm(matTheta[:,u_id])
    if norm_tmp >= 1
      matTheta[:,u_id] = matTheta[:,u_id] / norm_tmp - 10e-6 * sign.(matTheta[:,u_id])
    end
  end

  return nothing
end

function update_matBeta_poincare4(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                                  vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                  matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64, lr::Float64,
                                  usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                                  itr::Int64, mu::Float64, mu2::Float64, mt::Array{Float64,2}, nt::Array{Float64,2})

  update_matTheta_poincare4(matBeta, matTheta, vecBiasI, vecBiasU, matX_train', alpha, lambda, lr, itm_idx_len, usr_idx_len, itm_idx, usr_idx, itr, mu, mu2, mt, nt)
end


#
# Mini-batch Stochastic Poincare MF
#
function update_matTheta_poincare5(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                                   vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                   matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64, lr::Float64,
                                   usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                                   train_itr::Int64, mu::Float64, mu2::Float64, mt::Array{Float64,2}, nt::Array{Float64,2})

  sampled_i, sampled_j, sampled_v = findnz(matX_train[usr_idx, itm_idx])

  vec_sq_norm_theta = sum(matTheta[:, usr_idx].^2, 1)[:]
  vec_a = 1 .- vec_sq_norm_theta
  vec_sq_norm_beta = sum(matBeta[:, itm_idx].^2, 1)[:]
  vec_b = 1 .- vec_sq_norm_beta

  mat_partial_L_by_theta = zeros(Float64, size(matTheta,1), usr_idx_len)
  vec_partial_L_by_gamma = zeros(Float64, usr_idx_len)


  for u = 1:usr_idx_len
    u_id = usr_idx[u]

    # Compute vec_partial_L_by_d
    # size(itm_idx_len)
    tmp_theta = matTheta[:,u_id]
    sq_l2_dist_u = sum(broadcast(+, matBeta[:,itm_idx], -tmp_theta).^2, 1)[:]
    vec_f_ui = exp.(acosh.( 1 .+ 2 * sq_l2_dist_u ./ (vec_a[u] .* vec_b) ))
    vec_partial_L_by_d = ( 1. ./ (-1 .+ vec_f_ui) .- (1. .+ alpha .* sqrt.(matX_train[u_id, itm_idx])) ./ (1. .+ vec_f_ui) ) .* vec_f_ui

    # Compute vec_partial_d_by_theta
    # size(K, itm_idx_len)
    vec_c = 1 .+ 2 / vec_a[u] * sq_l2_dist_u ./ vec_b
    vec_partial_d_by_theta_u = tmp_theta * ((vec_sq_norm_beta' - 2 * (tmp_theta' * matBeta[:,itm_idx]) .+ 1) / vec_a[u]^2) .- matBeta[:,itm_idx] / vec_a[u]
    vec_partial_d_by_theta_u = broadcast(*, vec_partial_d_by_theta_u, 4 ./ (vec_b .* sqrt.(vec_c.^2 .- 1))')

    mat_partial_L_by_theta[:, u] += vec_partial_d_by_theta_u * vec_partial_L_by_d
  end

  println(mat_partial_L_by_theta[:,1])

  # Update matTheta by Adam (Adaptive Moment Estimation)
  # Kingma, D., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
  for u = 1:usr_idx_len
    u_id = usr_idx[u]
    gt = vec_a[u]^2 / 4 * mat_partial_L_by_theta[:,u] - lambda * matTheta[:,u_id]
    mt[:, u_id] = mu * mt[:, u_id] + (1 - mu) * gt
    nt[:, u_id] = mu2 * nt[:, u_id] + (1 - mu2) * (gt .* gt)
    mt[:, u_id] = mt[:, u_id] ./ (1 - mu ^ 1000)
    nt[:, u_id] = nt[:, u_id] ./ (1 - mu2 ^ 1000)
    matTheta[:,u_id] += lr * mt[:, u_id] ./ (sqrt.(nt[:, u_id]) .+ 10e-9)
    norm_tmp = norm(matTheta[:,u_id])
    if norm_tmp >= 1
      matTheta[:,u_id] = matTheta[:,u_id] / norm_tmp - 10e-6 * sign.(matTheta[:,u_id])
    end
  end

  return nothing
end

function update_matBeta_poincare5(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                                  vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                  matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64, lr::Float64,
                                  usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                                  itr::Int64, mu::Float64, mu2::Float64, mt::Array{Float64,2}, nt::Array{Float64,2})

  update_matTheta_poincare5(matBeta, matTheta, vecBiasI, vecBiasU, matX_train', alpha, lambda, lr, itm_idx_len, usr_idx_len, itm_idx, usr_idx, itr, mu, mu2, mt, nt)
end


function obj_val(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                 vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                 matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64,
                 usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})
  ret = 0

  vec_sq_norm_theta = sum(matTheta[:, usr_idx].^2, 1)[:]
  vec_a = 1 .- vec_sq_norm_theta
  vec_sq_norm_beta = sum(matBeta[:, itm_idx].^2, 1)[:]
  vec_b = 1 .- vec_sq_norm_beta

  for u = 1:usr_idx_len
    u_id = usr_idx[u]
    tmp_theta = matTheta[:,u_id]
    sq_l2_dist_u = sum(broadcast(+, matBeta[:,itm_idx], -tmp_theta).^2, 1)[:]
    vec_f_ui = exp.(acosh.( 1 .+ 2 * sq_l2_dist_u ./ (vec_a[u] .* vec_b) ))
    ret += sum(log.(vec_f_ui .- 1) - (1. .+ alpha * sqrt.(matX_train[u_id, itm_idx])) .* log.(1. .+ vec_f_ui))
  end

  ret -= 0.5 * lambda * sum(vec_sq_norm_theta)
  ret -= 0.5 * lambda * sum(vec_sq_norm_beta)

  return ret
end


#
# Logistic-Softmax-based objective function
#
function update_matTheta_poincare_logsoftmax(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                                   vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                   matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64, lr::Float64,
                                   usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                                   itr::Int64, mu::Float64, mu2::Float64, mt::Array{Float64,2}, nt::Array{Float64,2})

  sampled_i, sampled_j, sampled_v = findnz(matX_train[usr_idx, itm_idx])

  vec_sq_norm_theta = sum(matTheta[:, usr_idx].^2, 1)[:]
  vec_a = 1 .- vec_sq_norm_theta
  vec_sq_norm_beta = sum(matBeta[:, itm_idx].^2, 1)[:]
  vec_b = 1 .- vec_sq_norm_beta

  mat_partial_L_by_theta = zeros(Float64, size(matTheta,1), usr_idx_len)
  vec_partial_L_by_gamma = zeros(Float64, usr_idx_len)

  for itr=1:length(sampled_i)
    u = sampled_i[itr]
    i = sampled_j[itr]
    u_id = usr_idx[u]
    i_id = itm_idx[i]

    # Compute vec_partial_L_by_d
    sample_size = min(11, itm_idx_len-nnz(matX_train[u_id,:])+1)

    vec_sampled_neg_i_id = zeros(Int64, sample_size)
    neg_i_itr = 1
    while neg_i_itr < sample_size
      sampled_neg_i = rand(collect(1:itm_idx_len))
      sampled_neg_i_id = itm_idx[sampled_neg_i]
      if matX_train[u_id, sampled_neg_i_id] == 0 && length(find(vec_sampled_neg_i_id == sampled_neg_i)) == 0
        vec_sampled_neg_i_id[neg_i_itr] = sampled_neg_i
        neg_i_itr += 1
      end
    end
    vec_sampled_neg_i_id[neg_i_itr] = i

    tmp_theta = matTheta[:,u_id]
    tmp_beta = matBeta[:,itm_idx[vec_sampled_neg_i_id]]
    sq_l2_dist_ui = sum(broadcast(+, -tmp_beta, tmp_theta).^2, 1)[:]
    vec_f_ui = 1 ./ ( 1 .+ exp.(acosh.( 1 .+ 2 * sq_l2_dist_ui ./ (vec_a[u] .* vec_b[vec_sampled_neg_i_id]) )) )

    # Compute vec_partial_d_by_theta
    # size(K, sample_size + 1)
    vec_c = 1 .+ 2 / vec_a[u] ./ vec_b[vec_sampled_neg_i_id] .* sq_l2_dist_ui
    vec_partial_d_by_theta_u = tmp_theta * ((vec_sq_norm_beta[vec_sampled_neg_i_id]' - 2 * (tmp_theta' * tmp_beta) .+ 1) / vec_a[u]^2) .- tmp_beta / vec_a[u]
    vec_partial_d_by_theta_u = broadcast(*, vec_partial_d_by_theta_u, 4 ./ (vec_b[vec_sampled_neg_i_id] .* sqrt.(vec_c.^2 .- 1))')

    mat_partial_L_by_theta[:, u] += alpha * sqrt.(matX_train[u_id, i_id]) * (vec_f_ui[end] - 1) .* vec_partial_d_by_theta_u[:,end] -
                                    (vec_partial_d_by_theta_u * ((vec_f_ui .- 1) .* vec_f_ui)) / sum(vec_f_ui)
  end

  # Update matTheta by Adam (Adaptive Moment Estimation)
  # Kingma, D., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
  for u = 1:usr_idx_len
    u_id = usr_idx[u]
    gt = vec_a[u]^2 / 4 * mat_partial_L_by_theta[:,u] - lambda * matTheta[:,u_id]
    mt[:, u_id] = mu * mt[:, u_id] + (1 - mu) * gt
    nt[:, u_id] = mu2 * nt[:, u_id] + (1 - mu2) * (gt .* gt)
    mt[:, u_id] = mt[:, u_id] ./ (1 - mu ^ 600)
    nt[:, u_id] = nt[:, u_id] ./ (1 - mu2 ^ 600)
    matTheta[:,u_id] += lr * mt[:, u_id] ./ (sqrt.(nt[:, u_id]) .+ 10e-9)
    norm_tmp = norm(matTheta[:,u_id])
    if norm_tmp >= 1
      matTheta[:,u_id] = matTheta[:,u_id] / norm_tmp - 10e-6 * sign.(matTheta[:,u_id])
    end
  end

  return nothing
end

function update_matBeta_poincare_logsoftmax(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                                  vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                  matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64, lr::Float64,
                                  usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                                  itr::Int64, mu::Float64, mu2::Float64, mt::Array{Float64,2}, nt::Array{Float64,2})

  update_matTheta_poincare_logsoftmax(matBeta, matTheta, vecBiasI, vecBiasU, matX_train', alpha, lambda, lr, itm_idx_len, usr_idx_len, itm_idx, usr_idx, itr, mu, mu2, mt, nt)
end




function get_gradient_u(CS::Float64, alpha::Float64, matX_train_ui::SparseVector{Float64,Int64},
                        tmp_theta::Array{Float64,1}, tmp_beta::Array{Float64,2}, vecBiasU_uid::Float64, vecBiasI_iid::Array{Float64,1},
                        vec_sq_norm_theta_u::Float64, vec_sq_norm_beta::Array{Float64,1}, vec_a_u::Float64, vec_b::Array{Float64,1})
  # tmp_theta = matTheta[:,u_id]
  # tmp_beta = matBeta[:,itm_idx]
  # vec_sq_norm_theta_u = vec_sq_norm_theta[u]
  # vec_a_u = vec_a[u]
  # vecBiasU_uid = vecBiasU[u_id]
  # vecBiasI_iid = vecBiasI[itm_idx]
  # matX_train_ui = matX_train[u_id, itm_idx]

  # Compute vec_partial_L_by_d
  # size = (itm_idx_len)
  sq_l2_dist_ui = sum(broadcast(+, tmp_beta, -tmp_theta).^2, 1)[:]
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
  tmp_ui = tmp_theta * ( (vec_sq_norm_beta' - 2 * (tmp_theta' * tmp_beta) .+ 1) / vec_a_u^2 ) .- tmp_beta / vec_a_u
  tmp_ui = broadcast(*, tmp_ui, 4 ./ (vec_b .* sqrt.(vec_c_ui.^2 .- 1))')

  vec_c_u =  1 .+ 2 / vec_a_u * vec_sq_norm_theta_u
  tmp_u = (tmp_theta ./ vec_a_u^2) * (4 ./ sqrt.(vec_c_u.^2 .- 1)')

  ret = 2 * CS * broadcast(+, -broadcast(*, tmp_ui, vec_d_ui'), tmp_u * val_d_u) * vec_partial_L_by_d
  append!(ret, CS * sum(vec_partial_L_by_d))

  return ret
end
#
# Mini-batch Stochastic Poincare MF
#
function update_matTheta_poincare6(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                                   vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                   matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, CS::Float64, lambda::Float64, lr::Float64,
                                   usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                                   train_itr::Int64, mu::Float64, mu2::Float64, mt::Array{Float64,2}, nt::Array{Float64,2}, mtB::Array{Float64,1}, ntB::Array{Float64,1})

  sampled_i, sampled_j, sampled_v = findnz(matX_train[usr_idx, itm_idx])

  vec_sq_norm_theta = sum(matTheta[:, usr_idx].^2, 1)[:]
  vec_a = 1 .- vec_sq_norm_theta
  vec_sq_norm_beta = sum(matBeta[:, itm_idx].^2, 1)[:]
  vec_b = 1 .- vec_sq_norm_beta

  K = size(matTheta,1)
  #mat_partial_L_by_theta_biasU = SharedArray{Float64}(K+1, usr_idx_len)
  #vec_partial_L_by_biasU = SharedArray{Float64}(usr_idx_len)


  mat_partial_L_by_theta_biasU = @parallel hcat for u = 1:usr_idx_len
    get_gradient_u(CS, alpha, matX_train[usr_idx[u], itm_idx],
                   matTheta[:,usr_idx[u]], matBeta[:,itm_idx], vecBiasU[usr_idx[u]], vecBiasI[itm_idx],
                   vec_sq_norm_theta[u], vec_sq_norm_beta, vec_a[u], vec_b)
    # u_id = usr_idx[u]
    #
    # # Compute vec_partial_L_by_d
    # # size = (itm_idx_len)
    # tmp_theta = matTheta[:,u_id]
    # sq_l2_dist_ui = sum(broadcast(+, matBeta[:,itm_idx], -tmp_theta).^2, 1)[:]
    # val_d_u = acosh.( 1 .+ 2 * vec_sq_norm_theta[u] ./ vec_a[u] )
    # vec_d_i = acosh.( 1 .+ 2 * vec_sq_norm_beta ./ vec_b )
    # vec_d_ui = acosh.( 1 .+ 2 * sq_l2_dist_ui ./ vec_a[u] ./ vec_b )
    # vec_s_ui = CS * ( val_d_u.^2 .+ vec_d_i.^2 .- vec_d_ui.^2 + vecBiasU[u_id] + vecBiasI[itm_idx])
    # vec_f_ui = 1 ./ ( 1. .+ exp.(-vec_s_ui) )
    # vec_f_ui[isnan.(vec_f_ui)] = 1.
    # vec_partial_L_by_d = alpha * matX_train[u_id, itm_idx] .* (1 - vec_f_ui) - vec_f_ui
    #
    # # Compute vec_partial_d_by_theta
    # # size = (K, itm_idx_len)
    # vec_c_ui = 1 .+ 2 / vec_a[u] * sq_l2_dist_ui ./ vec_b
    # tmp_ui = tmp_theta * ( (vec_sq_norm_beta' - 2 * (tmp_theta' * matBeta[:,itm_idx]) .+ 1) / vec_a[u]^2 ) .- matBeta[:,itm_idx] / vec_a[u]
    # tmp_ui = broadcast(*, tmp_ui, 4 ./ (vec_b .* sqrt.(vec_c_ui.^2 .- 1))')
    #
    # vec_c_u =  1 .+ 2 / vec_a[u] * vec_sq_norm_theta[u]
    # tmp_u = (tmp_theta ./ vec_a[u]^2) * (4 ./ sqrt.(vec_c_u.^2 .- 1)')
    #
    # mat_partial_L_by_theta[:, u] += 2 * CS * broadcast(+, -broadcast(*, tmp_ui, vec_d_ui'), tmp_u * val_d_u) * vec_partial_L_by_d
    # vec_partial_L_by_biasU[u] = CS * sum(vec_partial_L_by_d)
  end

  # Update matTheta by Adam (Adaptive Moment Estimation)
  # Kingma, D., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
  for u = 1:usr_idx_len
    u_id = usr_idx[u]
    gt = vec_a[u]^2 / 4 * mat_partial_L_by_theta_biasU[1:K,u] - lambda * matTheta[:,u_id]
    mt[:, u_id] = mu * mt[:, u_id] + (1 - mu) * gt
    nt[:, u_id] = mu2 * nt[:, u_id] + (1 - mu2) * (gt .* gt)
    mt[:, u_id] = mt[:, u_id] ./ (1 - mu ^ train_itr)
    nt[:, u_id] = nt[:, u_id] ./ (1 - mu2 ^ train_itr)
    matTheta[:,u_id] += lr * mt[:, u_id] ./ (sqrt.(nt[:, u_id]) .+ 10e-9)
    norm_tmp = norm(matTheta[:,u_id])
    if norm_tmp >= 1
      matTheta[:,u_id] = matTheta[:,u_id] / norm_tmp - 10e-6 * sign.(matTheta[:,u_id])
    end

    gt = mat_partial_L_by_theta_biasU[K+1,u] - lambda * vecBiasU[u_id]
    mtB[u_id] = mu * mtB[u_id] + (1 - mu) * gt
    ntB[u_id] = mu2 * ntB[u_id] + (1 - mu2) * (gt .* gt)
    mtB[u_id] = mtB[u_id] ./ (1 - mu ^ train_itr)
    ntB[u_id] = ntB[u_id] ./ (1 - mu2 ^ train_itr)
    vecBiasU[u_id] = vecBiasU[u_id] + lr * mtB[u_id] ./ (sqrt(ntB[u_id]) + 10e-9)
  end

  return nothing
end

function update_matBeta_poincare6(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                                  vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                  matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, CS::Float64, lambda::Float64, lr::Float64,
                                  usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                                  itr::Int64, mu::Float64, mu2::Float64, mt::Array{Float64,2}, nt::Array{Float64,2}, mtB::Array{Float64,1}, ntB::Array{Float64,1})

  update_matTheta_poincare6(matBeta, matTheta, vecBiasI, vecBiasU, matX_train', alpha, CS, lambda, lr, itm_idx_len, usr_idx_len, itm_idx, usr_idx, itr, mu, mu2, mt, nt, mtB, ntB)
end


function obj_val6(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
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

  return ret / usr_idx_len / itm_idx_len
end

function PoincareDistance(vecA::Array{Float64,1}, vecB::Array{Float64,1})
  return acosh(1 + 2 * sum((vecA-vecB).^2) / (1-norm(vecA)^2) / (1-norm(vecB)^2) )
end

function sim_poincare(vecA::Array{Float64,1}, vecB::Array{Float64,1})
  return 0.5 * (PoincareDistance(vecA, zeros(length(vecA)))^2 + PoincareDistance(vecA, zeros(length(vecA)))^2 - PoincareDistance(vecA, vecB)^2)
end
