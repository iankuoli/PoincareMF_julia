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

  vec_norm_beta = zeros(itm_idx_len)
  vec_b = zeros(itm_idx_len)

  for i = 1:itm_idx_len
    vec_norm_beta[i] = norm(matBeta[itm_idx[i],:])
  end
  vec_b = 1 .- vec_norm_beta.^2

  @time @simd for u = 1:usr_idx_len
    u_id = usr_idx[u]

    # Compute partial L w.r.t distance
    norm_theta_u = norm(matTheta[u_id,:])
    a = 1 - norm_theta_u^2

    l2_dist_ui = zeros(itm_idx_len)
    @simd for i = 1:itm_idx_len
      l2_dist_ui[i] = norm(matTheta[u_id,:] - matBeta[itm_idx[i],:])
    end
    vec_f_ui = exp.(acosh.( 1 .+ 2 * l2_dist_ui.^2 ./ (a .* vec_b) ))
    vec_partial_L_by_d = 1 ./ (-1 .+ vec_f_ui) .- (1 .+ alpha .* matX_train[u_id, itm_idx]) ./ (1 .+ vec_f_ui) .* vec_f_ui

    # Compute partial distance w.r.t theta_u
    vec_c = 1 .+ 2 / a * (l2_dist_ui .^ 2) ./ vec_b
    vec_partial_d_by_theta_u = (vec_norm_beta.^2 - 2 * matBeta[itm_idx,:] * matTheta[u_id,:] .+ 1) / a^2 * matTheta[u_id,:]' .- matBeta[itm_idx,:] / a
    vec_partial_d_by_theta_u = broadcast(*, vec_partial_d_by_theta_u, 4 ./ (vec_b .* sqrt(vec_c.^2 .- 1)))

    # Update matTheta
    partial_theta = vec_partial_d_by_theta_u' * vec_partial_L_by_d
    matTheta[u_id,:] += lr * a^2 / 4 * partial_theta
    norm_tmp = norm(matTheta[u_id,:])
    if norm_tmp >= 1
      matTheta[u_id,:] = matTheta[u_id,:] / norm_tmp - 10e-6 * sign(matTheta[u_id,:])
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
