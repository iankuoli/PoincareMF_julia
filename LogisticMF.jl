include("evaluate.jl")
include("sample_data.jl")
include("LogisticMF_ParamUpdate.jl")
include("model_io.jl")


function LogisticMF(model_type::String, K::Int64, M::Int64, N::Int64,
                    matX_train::SparseMatrixCSC{Float64,Int64}, matX_test::SparseMatrixCSC{Float64,Int64}, matX_valid::SparseMatrixCSC{Float64,Int64},
                    ini_scale::Float64=0.003, alpha::Float64=1.0, lambda::Float64=0.0, lr::Float64=0.01, usr_batch_size::Int64=0, MaxItr::Int64=100,
                    topK::Array{Int64,1} = [10], test_step::Int64=0, check_step::Int64=5)
  #
  # Logistic Matrix Factorization for Implicit Feedback Data
  # In NIPS, 2014.
  #

  ## Parameters declaration
  usr_zeros = find((sum(matX_train, 2) .== 0)[:])
  itm_zeros = find((sum(matX_train, 1) .== 0)[:])

  IsConverge = false
  itr = 0

  valid_precision = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  valid_recall = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  #Vlog_likelihood = zeros(Float64, Int(ceil(MaxItr/check_step)))

  test_precision = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  test_recall = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  #Tlog_likelihood = zeros(Float64, Int(ceil(MaxItr/check_step)))


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

  l = 0;

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
    println(s)

    #
    # Update (matBeta, vecBiasU) & (matTheta & vecBiasU)
    #
    update_matBeta2(matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, lambda, K, usr_idx_len, itm_idx_len, usr_idx, itm_idx)
    update_matTheta2(matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, lambda, K, usr_idx_len, itm_idx_len, usr_idx, itm_idx)


    #
    # Validation
    #
    if mod(itr, check_step) == 0 && check_step > 0
      println("Validation ... ")
      indx = Int(itr / check_step)
      valid_precision[indx,:], test_recall[indx,:] = evaluateLogisticMF(matX_valid, matX_train, matTheta, vecBiasU, matBeta, vecBiasI, topK, 10., 1.)
      println("validation precision: " * string(valid_precision[indx,:]))
      println("validation recall: " * string(valid_recall[indx,:]))

      #
      # Check whether the step performs the best. If yes, run testing and save model
      #
      if findmax(valid_precision[:,1])[2] == Int(itr / check_step)
        # Testing
        println("Testing ... ")
        indx = Int(itr / check_step)
        test_precision[indx,:], test_recall[indx,:] = evaluateLogisticMF(matX_test, matX_train, matTheta, vecBiasU, matBeta, vecBiasI, topK, 10., 1.)
        println("testing precision: " * string(test_precision[indx,:]))
        println("testing recall: " * string(test_recall[indx,:]))

        # Save model
        file_name = string(model_type, "_K", K, "_", string(now())[1:10])
        write_model_PoincareMF(file_name, matTheta, matBeta, vecBiasU, vecBiasI, alpha, lr)
      end
    end


    #
    # Testing
    #
    if test_step > 0 && mod(itr, test_step) == 0
      println("Testing ... ")
      indx = Int(itr / test_step)
      test_precision[indx,:], test_recall[indx,:] = evaluateLogisticMF(matX_test, matX_train, matTheta, vecBiasU, matBeta, vecBiasI, topK, 10., 1.)
      println("testing precision: " * string(test_precision[indx,:]))
      println("testing recall: " * string(test_recall[indx,:]))
    end

    if test_step > 0 && mod(itr, test_step) == 0
      test_usr_idx, test_itm_idx, test_val = findnz(matX_test)
      test_usr_idx = unique(test_usr_idx)
      list_vecPrecision = zeros(length(topK))
      list_vecRecall = zeros(length(topK))
      step_size = 10000

      for j = 1:ceil(length(test_usr_idx)/step_size)
        range_step = Int(1 + (j-1) * step_size):Int(min(j*step_size, length(test_usr_idx)))

        # Compute the Precision and Recall
        test_matPredict = broadcast(+, broadcast(+, matTheta[test_usr_idx[range_step],:] * matBeta', vecBiasU[test_usr_idx[range_step]]), vecBiasI')
        test_matPredict -= test_matPredict .* (matX_train[test_usr_idx[range_step], :] .> 0)
        vec_precision, vec_recall = compute_precNrec(matX_test[test_usr_idx[range_step], :], test_matPredict, topK)
        list_vecPrecision = list_vecPrecision + sum(vec_precision, 1)[:]
        list_vecRecall = list_vecRecall + sum(vec_recall, 1)[:]
      end

      avg_test_precision = list_vecPrecision / length(test_usr_idx)
      avg_test_recall = list_vecRecall / length(test_usr_idx)

      println(list_vecPrecision)
      println(list_vecRecall)
      println(length(test_usr_idx))

      println("testing precision: " * string(avg_test_precision))
      println("testing recall: " * string(avg_test_recall))
    end
  end

  return test_precision, test_recall, valid_precision, valid_recall, matTheta, vecBiasU, matBeta, vecBiasI
end
