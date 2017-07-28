include("maxK.jl")

function inference(usr_idx::Array{Int64,1}, matTheta::Array{Float64,2}, matBeta::Array{Float64,2})
  return matTheta[usr_idx,:] * matBeta';
end

function inferenceLogisticMF(usr_idx::Array{Int64,1}, matTheta::Array{Float64,2}, vecBiasU::Array{Float64,1}, matBeta::Array{Float64,2}, vecBiasI::Array{Float64,1})
  return broadcast(+, broadcast(+, matTheta[usr_idx,:] * matBeta', vecBiasU[usr_idx]), vecBiasI')
end


function inference_Poincare_sqdist(usr_idx::Array{Int64,1}, matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1})
  ret = zeros(length(usr_idx), size(matBeta, 2))

  vec_sq_norm_theta = sum(matTheta[:,usr_idx] .^ 2, 1)[:]
  vec_sq_norm_beta = sum(matBeta .^ 2, 1)[:]

  for u = 1:length(usr_idx)
    tmp1 = acosh.( 1 .+ 2 .* sum(broadcast(-, matBeta, matTheta[:,usr_idx[u]]) .^ 2, 1)[:] ./ ((1 - vec_sq_norm_theta[u]) .* (1 .- vec_sq_norm_beta)) )
    ret[u,:] = -tmp1.^2 + vecBiasU[usr_idx[u]] + vecBiasI
  end

  return ret
end


function inference_Poincare2(usr_idx::Array{Int64,1}, matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1})
  ret = zeros(length(usr_idx), size(matBeta, 2))

  vec_sq_norm_theta = sum(matTheta[:,usr_idx] .^ 2, 1)[:]
  vec_sq_norm_beta = sum(matBeta .^ 2, 1)[:]

  for u = 1:length(usr_idx)
    tmp1 = acosh.( 1 .+ 2 .* sum(broadcast(-, matBeta, matTheta[:,usr_idx[u]]) .^ 2, 1)[:] ./ ((1 - vec_sq_norm_theta[u]) .* (1 .- vec_sq_norm_beta)) )
    tmp2 = acosh.( 1 .+ 2 .* sum(matBeta.^2, 1)[:] ./ (1 .- vec_sq_norm_beta) )
    tmp3 = acosh.( 1 .+ 2 .* sum(matTheta[:,usr_idx[u]].^2, 1)[:] ./ (1 - vec_sq_norm_theta[u]) )
    ret[u,:] = tmp2.^2 .+ tmp3.^2 .- tmp1.^2 + vecBiasU[usr_idx[u]] + vecBiasI
  end

  return ret
end


function inference_Poincare_tfidf(log_sum_X_i::Array{Float64,1}, usr_idx::Array{Int64,1}, matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1})
  ret = zeros(length(usr_idx), size(matBeta, 2))

  vec_sq_norm_theta = sum(matTheta[:,usr_idx] .^ 2, 1)[:]
  vec_sq_norm_beta = sum(matBeta .^ 2, 1)[:]

  for u = 1:length(usr_idx)
    tmp1 = acosh.( 1 .+ 2 .* sum(broadcast(-, matBeta, matTheta[:,usr_idx[u]]) .^ 2, 1)[:] ./ ((1 - vec_sq_norm_theta[u]) .* (1 .- vec_sq_norm_beta)) )
    tmp2 = acosh.( 1 .+ 2 .* sum(matBeta.^2, 1)[:] ./ (1 .- vec_sq_norm_beta) )
    tmp3 = acosh.( 1 .+ 2 .* sum(matTheta[:,usr_idx[u]].^2, 1)[:] ./ (1 - vec_sq_norm_theta[u]) )
    ret[u,:] = (tmp2.^2 .+ tmp3.^2 .- tmp1.^2 + vecBiasU[usr_idx[u]] + vecBiasI) .* log_sum_X_i
  end

  return ret
end


function inference_Poincare_inverse(usr_idx::Array{Int64,1}, matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1})
  ret = zeros(length(usr_idx), size(matBeta, 2))

  vec_sq_norm_theta = sum(matTheta[:,usr_idx] .^ 2, 1)[:]
  vec_sq_norm_beta = sum(matBeta .^ 2, 1)[:]

  for u = 1:length(usr_idx)
    tmp1 = acosh.( 1 .+ 2 .* sum(broadcast(-, matBeta, matTheta[:,usr_idx[u]]) .^ 2, 1)[:] ./ ((1 - vec_sq_norm_theta[u]) .* (1 .- vec_sq_norm_beta)) )
    tmp2 = acosh.( 1 .+ 2 .* sum(matBeta.^2, 1)[:] ./ (1 .- vec_sq_norm_beta) )
    tmp3 = acosh.( 1 .+ 2 .* sum(matTheta[:,usr_idx[u]].^2, 1)[:] ./ (1 - vec_sq_norm_theta[u]) )
    ret[u,:] = (1 .- tmp2).^2 .+ (1 .- tmp3).^2 .- tmp1.^2 + vecBiasU[usr_idx[u]] + vecBiasI
  end

  return ret
end


function infer_entry(matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, i_idx, j_idx)
  return (matTheta[i_idx,:]' * matBeta[j_idx, :])[1]
end
