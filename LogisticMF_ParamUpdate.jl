function update_matTheta_default(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                                 vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                 alpha::Float64, lambda::Float64, K::Int64,
                                 usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})
  #
  # Update matTheta & vecBiasU
  #
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

  return matTheta, matBeta, vecBiasU, vecBiasI
end


function update_matBeta_default(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                                vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                                alpha::Float64, lambda::Float64, K::Int64,
                                usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})

  return update_matTheta_default(matTheta, matBeta, vecBiasU, vecBiasI, alpha, lambda, K, usr_idx_len, itm_idx_len, usr_idx, itm_idx)
end
