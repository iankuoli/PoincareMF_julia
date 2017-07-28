function update_matTheta(alpha::Float64, matX_train::SparseMatrixCSC{Float64,Int64},
                         matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                         usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                         train_itr::Int64, lr::Float64, lambda::Float64, mu::Float64, mu2::Float64,
                         mt::Array{Float64,2}, nt::Array{Float64,2}, mtB::Array{Float64,1}, ntB::Array{Float64,1})

  # Compute partial L w.r.t distance
  predict_X = broadcast(+, broadcast(+, matTheta[usr_idx,:] * matBeta[itm_idx,:]', vecBiasU[usr_idx]), vecBiasI[itm_idx]')
  matTmp = 1. ./ (1. + exp.(-predict_X))
  matTmp[isnan.(matTmp)] = 1.
  matTmp = alpha * matX_train[usr_idx, itm_idx] .* (1. - matTmp) - matTmp
  #matTmp = alpha * matX_train[usr_idx, itm_idx] - (1. + alpha * matX_train[usr_idx, itm_idx]) .* matTmp

  # Update matTheta
  partial_theta = matTmp * matBeta[itm_idx, :] - lambda * matTheta[usr_idx,:]
  #matTheta[usr_idx,:] += lr * partial_theta
  #grad_sqr_sum_theta += partial_theta .^ 2
  mt[usr_idx,:] = mu * mt[usr_idx,:] + (1 - mu) * partial_theta
  nt[usr_idx,:] = mu2 * nt[usr_idx,:] + (1 - mu2) * (partial_theta .* partial_theta)
  mt[usr_idx,:] = mt[usr_idx,:] ./ (1 - mu ^ train_itr)
  nt[usr_idx,:] = nt[usr_idx,:] ./ (1 - mu2 ^ train_itr)
  matTheta[usr_idx,:] += lr * mt[usr_idx,:] ./ (sqrt.(nt[usr_idx,:]) .+ 10e-9)


  # Update vecBiasU
  partial_biasU = sum(matTmp, 2)[:]
  #vecBiasU[usr_idx] += lr * partial_biasU
  #grad_sqr_sum_biasU += partial_biasU .^ 2
  mtB[usr_idx] = mu * mtB[usr_idx] + (1 - mu) * partial_biasU
  ntB[usr_idx] = mu2 * ntB[usr_idx] + (1 - mu2) * (partial_biasU .* partial_biasU)
  mtB[usr_idx] = mtB[usr_idx] ./ (1 - mu ^ train_itr)
  ntB[usr_idx] = ntB[usr_idx] ./ (1 - mu2 ^ train_itr)
  vecBiasU[usr_idx] += lr * mtB[usr_idx] ./ (sqrt.(ntB[usr_idx]) .+ 10e-9)

  return nothing
end


function update_matBeta(alpha::Float64, matX_train::SparseMatrixCSC{Float64,Int64},
                        matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                        usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                        train_itr::Int64, lr::Float64, lambda::Float64, mu::Float64, mu2::Float64,
                        mt::Array{Float64,2}, nt::Array{Float64,2}, mtB::Array{Float64,1}, ntB::Array{Float64,1})

  update_matTheta(alpha, matX_train', matBeta, matTheta, vecBiasI, vecBiasU, itm_idx_len, usr_idx_len, itm_idx, usr_idx, train_itr, lr, lambda, mu, mu2, mt, nt, mtB, ntB)
end


function update_matTheta2(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                          vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                          matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64, K::Int64,
                          usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})

  for u = 1:usr_idx_len
    u_id = usr_idx[u]

    # Compute partial L w.r.t distance
    predict_X = broadcast(+, broadcast(+, (matBeta[itm_idx,:] * matTheta[u_id,:])' , vecBiasU[u_id]), vecBiasI[itm_idx]')[:]
    matTmp = exp.(predict_X)'
    matTmp = matTmp ./ (1 + matTmp)

    matTmp[isnan.(matTmp)] = 1
    matTmp = (1 + alpha * matX_train[u_id, itm_idx]) .* matTmp
    matTmp = alpha * matX_train[u_id, itm_idx] - matTmp

    # Update matTheta
    partial_theta = (matBeta[itm_idx, :]' * matTmp)[:] - 0.5 * lambda * matTheta[u_id,:]
    matTheta[u_id,:] += lr * partial_theta

    # Update vecBiasU
    partial_biasU = sum(matTmp)
    vecBiasU[u_id] += lr * partial_biasU
  end

  return nothing
end


function update_matBeta2(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                         vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                         matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64, K::Int64,
                         usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})

  update_matTheta2(matBeta, matTheta, vecBiasI, vecBiasU, matX_train', alpha, lambda, K, itm_idx_len, usr_idx_len, itm_idx, usr_idx)
end

#
# Logistic-based MF
#
function update_matTheta_logistic1(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
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
    vec_f_ui = exp.( -tmp_theta' *  tmp_beta)'
    vec_partial_L_by_d = ( 1. ./ (vec_f_ui) .- (1. .+ alpha .* matX_train[u_id, itm_idx[vec_sampled_neg_i_id]]) ./ (1. .+ vec_f_ui) ) .* vec_f_ui

    mat_partial_L_by_theta[:, u] += -tmp_beta * vec_partial_L_by_d
  end

  # Update matTheta by Adam (Adaptive Moment Estimation)
  # Kingma, D., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
  for u = 1:usr_idx_len
    u_id = usr_idx[u]
    gt = mat_partial_L_by_theta[:,u] - lambda * matTheta[:,u_id]
    # mt[:, u_id] = mu * mt[:, u_id] + (1 - mu) * gt
    # nt[:, u_id] = mu2 * nt[:, u_id] + (1 - mu2) * (gt .* gt)
    # mt[:, u_id] = mt[:, u_id] ./ (1 - mu ^ itr)
    # nt[:, u_id] = nt[:, u_id] ./ (1 - mu2 ^ itr)
    # matTheta[:,u_id] += lr * mt[:, u_id] ./ (sqrt.(nt[:, u_id]) .+ 10e-9)
    matTheta[:,u_id] += lr * gt
  end

  return nothing
end

function update_matBeta_logistic1(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                                  vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                  matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64, lr::Float64,
                                  usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                                  itr::Int64, mu::Float64, mu2::Float64, mt::Array{Float64,2}, nt::Array{Float64,2})

  update_matTheta_logistic1(matBeta, matTheta, vecBiasI, vecBiasU, matX_train', alpha, lambda, lr, itm_idx_len, usr_idx_len, itm_idx, usr_idx, itr, mu, mu2, mt, nt)
end


function obj_LogMF_val(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                       vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                       matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64,
                       usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})
  ret = 0
  for u = 1:usr_idx_len
    u_id = usr_idx[u]
    vec_f_ui = 1 ./ ( 1 .+ exp.(-( matBeta[itm_idx,:] * matTheta[u_id,:] + vecBiasU[u_id] + vecBiasI[itm_idx])) )
    vec_f_ui[isnan.(vec_f_ui)] = 1
    ret += sum(alpha * matX_train[u_id, itm_idx] .* log.(vec_f_ui .+ 10e-20) + log.(1. .- vec_f_ui .+ 10e-20))
  end

  return ret / usr_idx_len / itm_idx_len
end
