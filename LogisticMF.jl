include("evaluate.jl")
include("sample_data.jl")
include("LogisticMF_ParamUpdate.jl")
include("model_io.jl")


function LogisticMF(model_type::String, K::Int64, M::Int64, N::Int64,
                    matX_train::SparseMatrixCSC{Float64,Int64}, matX_test::SparseMatrixCSC{Float64,Int64}, matX_valid::SparseMatrixCSC{Float64,Int64},
                    ini_scale::Float64=0.003, alpha::Float64=1.0, lambda::Float64=0.0, lr::Float64=0.01, usr_batch_size::Int64=0, MaxItr::Int64=100,
                    topK::Array{Int64,1} = [10], test_step::Int64=0, check_step::Int64=5, mu::Float64=0.2, mu2::Float64=0.4)
  #
  # Logistic Matrix Factorization for Implicit Feedback Data
  # In NIPS, 2014.
  #

  ## Parameters declaration
  usr_zeros = find((sum(matX_train, 2) .== 0)[:])
  itm_zeros = find((sum(matX_train, 1) .== 0)[:])

  IsConverge = false
  itr = 0

  train_precision = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  train_recall = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))

  valid_precision = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  valid_recall = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  #Vlog_likelihood = zeros(Float64, Int(ceil(MaxItr/check_step)))

  test_precision = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  test_recall = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  #Tlog_likelihood = zeros(Float64, Int(ceil(MaxItr/check_step)))

  bestTheta = zeros(K, M)
  bestBeta = zeros(K, N)
  bestBiasU = zeros(M)
  bestBiasI = zeros(N)

  #
  # Initialization
  #
  # Initialize matTheta
  matTheta = ini_scale * rand(M, K)
  matTheta[usr_zeros,:] = 0

  # Initialize matBeta
  matBeta = ini_scale * rand(N, K)
  matBeta[itm_zeros, :] = 0

  #Initialize Biases
  vecBiasU = ini_scale * rand(M) + 0.3
  vecBiasU[usr_zeros] = 0
  vecBiasI = ini_scale * rand(N) + 0.3
  vecBiasI[itm_zeros] = 0

  if usr_batch_size == M || usr_batch_size == 0
    usr_idx, itm_idx, usr_idx_len, itm_idx_len = sample_data(M, N, usr_batch_size, matX_train, usr_zeros, itm_zeros)
    sampled_i, sampled_j, sampled_v = findnz(matX_train[usr_idx, itm_idx])
  end

  mt_theta = zeros(size(matTheta'))
  nt_theta = zeros(size(matTheta'))
  mt_beta = zeros(size(matBeta'))
  nt_beta = zeros(size(matBeta'))

  vecBiasU = zeros(M)
  vecBiasI = zeros(N)

  objval = obj_LogMF_val(matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, lambda, usr_idx_len, itm_idx_len, usr_idx, itm_idx)
  s = @sprintf "     Objective Function at step %d: %f" itr objval
  println(s)

  mt_theta = zeros(size(matTheta))
  nt_theta = zeros(size(matTheta))
  mt_beta = zeros(size(matBeta))
  nt_beta = zeros(size(matBeta))
  mt_biasU = zeros(size(vecBiasU))
  nt_biasU = zeros(size(vecBiasU))
  mt_biasI = zeros(size(vecBiasI))
  nt_biasI = zeros(size(vecBiasI))

  while IsConverge == false && itr < MaxItr
    itr += 1

    ## Update the latent parameters

    #
    # Sample data
    #
    if usr_batch_size != M && usr_batch_size != 0
      usr_idx, itm_idx, usr_idx_len, itm_idx_len = sample_data(M, N, usr_batch_size, matX_train, usr_zeros, itm_zeros)
      sampled_i, sampled_j, sampled_v = findnz(matX_train[usr_idx, itm_idx])
    end

    grad_sqr_sum_theta = zeros(usr_idx_len, K)
    grad_sqr_sum_biasU = zeros(usr_idx_len)
    grad_sqr_sum_beta = zeros(itm_idx_len, K)
    grad_sqr_sum_biasI = zeros(itm_idx_len)

    s = @sprintf "Index: %d  ---------------------------------- %d , %d , lr: %f" itr usr_idx_len itm_idx_len lr
    print(s)

    #
    # Update (matBeta, vecBiasU) & (matTheta & vecBiasU)
    #
    update_matBeta(alpha, matX_train, matTheta, matBeta, vecBiasU, vecBiasI, usr_idx_len, itm_idx_len, usr_idx, itm_idx, itr, lr, lambda, mu, mu2, mt_beta, nt_beta, mt_biasI, nt_biasI)
    update_matTheta(alpha, matX_train, matTheta, matBeta, vecBiasU, vecBiasI, usr_idx_len, itm_idx_len, usr_idx, itm_idx, itr, lr, lambda, mu, mu2, mt_theta, nt_theta, mt_biasU, nt_biasU)

    #update_matBeta(matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, lambda, K, usr_idx_len, itm_idx_len, usr_idx, itm_idx)
    #update_matTheta(matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, lambda, K, usr_idx_len, itm_idx_len, usr_idx, itm_idx)

    objval = obj_LogMF_val(matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, lambda, usr_idx_len, itm_idx_len, usr_idx, itm_idx)

    s = @sprintf "     Objective Function at step %d: %f" itr objval
    println(s)


    #
    # Training Precision & Recall
    #
    if mod(itr, check_step) == 0 && check_step > 0
      println("Testing training data ... ")
      indx = Int(itr / check_step)

      (vec_usr_idx, j, v) = findnz(sum(matX_train, 2))
      list_vecPrecision = zeros(length(topK))
      list_vecRecall = zeros(length(topK))
      log_likelihood = 0
      step_size = 300
      denominator = 0

      ret_tmp = @parallel (+) for j = 1:ceil(Int64, length(vec_usr_idx)/step_size)
        infer_N_eval_LogisticMF(matX_train, matX_train, matTheta, vecBiasU, matBeta, vecBiasI, alpha, topK, vec_usr_idx, j, step_size, false)
      end

      sum_vecPrecision = ret_tmp[1:length(topK)]
      sum_vecRecall = ret_tmp[(length(topK)+1):2*length(topK)]
      sum_denominator = ret_tmp[end]

      train_precision[indx,:] = sum_vecPrecision / sum_denominator
      train_recall[indx,:] = sum_vecRecall / sum_denominator

      println("training precision: " * string(train_precision[indx,:]))
      println("training recall: " * string(train_recall[indx,:]))
    end


    #
    # Validation
    #
    if mod(itr, check_step) == 0 && check_step > 0
      println("Validation ... ")
      indx = Int(itr / check_step)
      valid_precision[indx,:], valid_recall[indx,:] = evaluateLogisticMF(matX_valid, matX_train, matTheta, vecBiasU, matBeta, vecBiasI, topK, 1.)
      println("validation precision: " * string(valid_precision[indx,:]))
      println("validation recall: " * string(valid_recall[indx,:]))

      #
      # Check whether the step performs the best. If yes, run testing and save model
      #
      if findmax(valid_precision[:,1])[2] == Int(itr / check_step)
        # Testing
        println("Testing ... ")
        indx = Int(itr / check_step)
        test_precision[indx,:], test_recall[indx,:] = evaluateLogisticMF(matX_test, matX_train, matTheta, vecBiasU, matBeta, vecBiasI, topK, 1.)
        println("testing precision: " * string(test_precision[indx,:]))
        println("testing recall: " * string(test_recall[indx,:]))

        # Save model
        file_name = string(model_type, "_K", K, "_", string(now())[1:10])
        write_model_PoincareMF(file_name, matTheta, matBeta, vecBiasU, vecBiasI, alpha, lr)

        bestTheta = copy(matTheta)
        bestBeta = copy(matBeta)
        bestBiasU = copy(vecBiasU)
        bestBiasI = copy(vecBiasI)
      end
    end


    #
    # Testing
    #
    if test_step > 0 && mod(itr, test_step) == 0
      println("Testing ... ")
      indx = Int(itr / test_step)
      test_precision[indx,:], test_recall[indx,:] = evaluateLogisticMF(matX_test, matX_train, matTheta, vecBiasU, matBeta, vecBiasI, topK, 1.)
      println("testing precision: " * string(test_precision[indx,:]))
      println("testing recall: " * string(test_recall[indx,:]))
    end
  end

  return test_precision, test_recall, valid_precision, valid_recall, train_precision, train_recall, bestTheta, bestBiasU, bestBeta, bestBiasI, matTheta, vecBiasU, matBeta, vecBiasI
end
