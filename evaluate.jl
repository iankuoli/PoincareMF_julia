include("measure.jl")
include("inference.jl")

function infer_N_eval_PRPF(matX::SparseMatrixCSC{Float64, Int64}, matX_train::SparseMatrixCSC{Float64, Int64},
                           matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, C::Float64, alpha::Float64,
                           topK::Array{Int64,1}, vec_usr_idx::Array{Int64,1}, j::Int64, step_size::Int64)

  range_step = collect((1 + (j-1) * step_size):min(j*step_size, length(vec_usr_idx)))

  # Compute the Precision and Recall
  matPredict = inference(vec_usr_idx[range_step], matTheta, matBeta)
  tmp_mask = sparse(findn(matX_train[vec_usr_idx[range_step], :])...,
                    ones(nnz(matX_train[vec_usr_idx[range_step], :])),
                    size(matX_train[vec_usr_idx[range_step], :])...)

  matPredict -= matPredict .* tmp_mask
  (vec_precision, vec_recall) = compute_precNrec(matX[vec_usr_idx[range_step], :], matPredict, topK)

  vecPrecision = sum(vec_precision, 1)[:]
  vecRecall = sum(vec_recall, 1)[:]
  log_likelihood =  LogPRPFObjFunc(C, alpha, matX[vec_usr_idx[range_step], :], matPredict) +
                     DistributionPoissonLogNZ(matX[vec_usr_idx[range_step], :], matPredict)
  denominator = length(range_step)

  return vcat(vecPrecision, vecRecall, log_likelihood, denominator)
end


function infer_N_eval_Pointcare(matX::SparseMatrixCSC{Float64, Int64}, matX_train::SparseMatrixCSC{Float64, Int64},
                                matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, vecGamma::Array{Float64,1}, vecDelta::Array{Float64,1},
                                topK::Array{Int64,1}, vec_usr_idx::Array{Int64,1}, j::Int64, step_size::Int64)

  range_step = collect((1 + (j-1) * step_size):min(j*step_size, length(vec_usr_idx)))

  # Compute the Precision and Recall
  matPredict = inference_Poincare(vec_usr_idx[range_step], matTheta, matBeta, vecGamma, vecDelta)
  tmp_mask = sparse(findn(matX_train[vec_usr_idx[range_step], :])...,
                    ones(nnz(matX_train[vec_usr_idx[range_step], :])),
                    size(matX_train[vec_usr_idx[range_step], :])...)

  matPredict -= matPredict .* tmp_mask
  (vec_precision, vec_recall) = compute_precNrec(matX[vec_usr_idx[range_step], :], matPredict, topK)

  vecPrecision = sum(vec_precision, 1)[:]
  vecRecall = sum(vec_recall, 1)[:]
  denominator = length(range_step)

  return vcat(vecPrecision, vecRecall, denominator)
end


function evaluatePoincareMF(matX::SparseMatrixCSC{Float64, Int64}, matX_train::SparseMatrixCSC{Float64, Int64},
                            matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, vecGamma::Array{Float64,1}, vecDelta::Array{Float64,1},
                            topK::Array{Int64,1}, alpha::Float64)

  (vec_usr_idx, j, v) = findnz(sum(matX, 2))
  list_vecPrecision = zeros(length(topK))
  list_vecRecall = zeros(length(topK))
  log_likelihood = 0
  step_size = 300
  denominator = 0

  println("QQ")
  ret_tmp = @parallel (+) for j = 1:ceil(Int64, length(vec_usr_idx)/step_size)
    infer_N_eval_Pointcare(matX, matX_train, matTheta, matBeta, vecGamma, vecDelta, topK, vec_usr_idx, j, step_size)
  end

  println("QQ")
  sum_vecPrecision = ret_tmp[1:length(topK)]
  sum_vecRecall = ret_tmp[(length(topK)+1):2*length(topK)]
  sum_denominator = ret_tmp[end]

  precision = sum_vecPrecision / sum_denominator
  recall = sum_vecRecall / sum_denominator

  return precision, recall
end


function evaluatePRPF(matX::SparseMatrixCSC{Float64, Int64}, matX_train::SparseMatrixCSC{Float64, Int64},
                  matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, topK::Array{Int64,1}, C::Float64, alpha::Float64)

  (vec_usr_idx, j, v) = findnz(sum(matX, 2))
  list_vecPrecision = zeros(length(topK))
  list_vecRecall = zeros(length(topK))
  log_likelihood = 0
  step_size = 300
  denominator = 0

  println("QQ")
  ret_tmp = @parallel (+) for j = 1:ceil(Int64, length(vec_usr_idx)/step_size)
    infer_N_eval_PRPF(matX, matX_train, matTheta, matBeta, C, alpha, topK, vec_usr_idx, j, step_size)
  end

  println("QQ")
  sum_vecPrecision = ret_tmp[1:length(topK)]
  sum_vecRecall = ret_tmp[(length(topK)+1):2*length(topK)]
  sum_log_likelihood = ret_tmp[end-1]
  sum_denominator = ret_tmp[end]

  precision = sum_vecPrecision / sum_denominator
  recall = sum_vecRecall / sum_denominator
  log_likelihood = sum_log_likelihood / countnz(matX)

  return precision, recall, log_likelihood
end


#  /// --- Unit test for function: evaluate() --- ///
#
# X =  sparse([5. 4 3 0 0 0 0 0;
#              3. 4 5 0 0 0 0 0;
#              0  0 0 3 3 4 0 0;
#              0  0 0 5 4 5 0 0;
#              0  0 0 0 0 0 5 4;
#              0  0 0 0 0 0 3 4])
# A = [1. 0 0; 1 0 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1]
# B = [4. 0 0; 4 0 0; 4 0 0; 0 4 0; 0 3 0; 0 5 0; 0 0 4; 0 0 4]
# XX = spzeros(6,8)
# topK = [1, 2]
# C = 1.
# alpha = 1000.
# precision, recall, log_likelihood = evaluate(X, XX, A, B, topK, C, alpha)


# theta1 = readdlm("/Users/iankuoli/Downloads/theta1.csv", ',')
# beta1 = readdlm("/Users/iankuoli/Downloads/beta1.csv", ',')
# precision, recall, likelihood = evaluate(matX_test, matX_train, theta1, beta1, topK, C, alpha)
#
# if precision == 0.629787
#   println("right")
# end


# using HDF5
#
# beta1 = h5read("/Users/iankuoli/Downloads/beta1.h5", "/dataset1")
# theta1 = h5read("/Users/iankuoli/Downloads/theta1.h5", "/dataset1")
#
# precision, recall, likelihood = evaluate(matX_test, matX_train, theta1, beta1, topK, C, alpha)
#
