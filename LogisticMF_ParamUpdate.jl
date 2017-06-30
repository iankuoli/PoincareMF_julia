function update_matTheta(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                         vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                         matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64, K::Int64,
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
                        matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64, K::Int64,
                        usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})

  update_matTheta(matBeta, matTheta, vecBiasI, vecBiasU, matX_train', alpha, lambda, K, itm_idx_len, usr_idx_len, itm_idx, usr_idx)
end


function update_matTheta2(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                          vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                          matX_train::SparseMatrixCSC{Float64,Int64}, alpha::Float64, lambda::Float64, K::Int64,
                          usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})

  for u = 1:usr_idx_len
    u_id = usr_idx[u]

    # Compute partial L w.r.t distance
    predict_X = broadcast(+, broadcast(+, (matBeta[itm_idx,:] * matTheta[u_id,:])' , vecBiasU[u_id]), vecBiasI[itm_idx]')[:]
    matTmp = exp(predict_X)
    matTmp = matTmp ./ (1 + matTmp)
    matTmp[isnan(matTmp)] = 1
    matTmp = (1 + alpha * matX_train[u_id, itm_idx]) .* matTmp
    matTmp = alpha * matX_train[u_id, itm_idx] - matTmp

    # Update matTheta
    partial_theta = (matTmp' * matBeta[itm_idx, :])[:] - 0.5 * lambda * matTheta[u_id,:]
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
