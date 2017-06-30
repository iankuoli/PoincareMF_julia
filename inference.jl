include("maxK.jl")

function inference(usr_idx::Array{Int64,1}, matTheta::Array{Float64,2}, matBeta::Array{Float64,2})
  return matTheta[usr_idx,:] * matBeta';
end

function inferenceLogisticMF(usr_idx::Array{Int64,1}, matTheta::Array{Float64,2}, vecBiasU::Array{Float64,1}, matBeta::Array{Float64,2}, vecBiasI::Array{Float64,1})
  return broadcast(+, broadcast(+, matTheta[usr_idx,:] * matBeta', vecBiasU[usr_idx]), vecBiasI')
end


function inference_Poincare(usr_idx::Array{Int64,1}, matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, vecGamma::Array{Float64,1}, vecDelta::Array{Float64,1})
  ret = zeros(length(usr_idx), size(matBeta, 1))
  vec_norm_beta = zeros(size(matBeta, 1))
  for itr = 1:size(matBeta,1)
    vec_norm_beta[itr] = norm(matBeta[itr,:])
  end

  for u = 1:length(usr_idx)
    u_id = usr_idx[u]
    norm_theta = norm(matTheta[u_id,:])

    for i_id = 1:size(matBeta,1)
      ret[u, i_id] = acosh( 1 + 2 * norm(matTheta[u_id,:] - matBeta[i_id,:])^2 / ( (1-norm_theta^2) * (1-vec_norm_beta[i_id]^2) ) )
    end
  end


  return exp.(-ret)
end

function infer_entry(matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, i_idx, j_idx)
  return (matTheta[i_idx,:]' * matBeta[j_idx, :])[1]
end
