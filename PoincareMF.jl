include("evaluate.jl")
include("sample_data.jl")
include("PoincareMF_ParamUpdate.jl")
include("model_io.jl")


function PoincareMF(Ini::Bool, probDropout::Float64, model_type::String, K::Int64, M::Int64, N::Int64,
                    matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, vecBiasU::Array{Float64,1}, vecBiasI::Array{Float64,1},
                    matX_train::SparseMatrixCSC{Float64,Int64}, matX_test::SparseMatrixCSC{Float64,Int64}, matX_valid::SparseMatrixCSC{Float64,Int64},
                    ini_scale::Float64=0.003, alpha::Float64=1.0, CS::Float64=10.0, lambda::Float64=0.0, lr::Float64=0.01, mu::Float64=0.6, mu2::Float64=0.8,
                    usr_batch_size::Int64=0, MaxItr::Int64=100, topK::Array{Int64,1} = [10], test_step::Int64=0, check_step::Int64=5)
  #
  # MF based on Poincare distance
  # next work.
  #

  ## Parameters declaration
  usr_zeros = find((sum(matX_train, 2) .== 0)[:])
  itm_zeros = find((sum(matX_train, 1) .== 0)[:])

  train_precision = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  train_recall = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))

  valid_precision = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  valid_recall = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))

  test_precision = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  test_recall = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  train_loglikelihood = zeros(Float64, MaxItr)

  bestTheta = zeros(K, M)
  bestBeta = zeros(K, N)
  bestBiasU = zeros(M)
  bestBiasI = zeros(N)

  #
  # Initialization
  #
  if Ini == true
    # Initialize matTheta
    matTheta = ini_scale * (rand(K, M) .- 0.5)
    matTheta[:,usr_zeros] = 0

    # Initialize matBeta
    matBeta = ini_scale * (rand(K, N) .- 0.5)
    matBeta[:,itm_zeros] = 0

    #Initialize Biases
    vecBiasU = 0.01 / K * rand(M)
    vecBiasU[usr_zeros] = 0
    vecBiasI = 0.01 / K * rand(N)
    vecBiasI[itm_zeros] = 0
  end

  # Epoch-wise Update
  if usr_batch_size == M || usr_batch_size == 0
    usr_idx, itm_idx, usr_idx_len, itm_idx_len = sample_data(M, N, usr_batch_size, matX_train, usr_zeros, itm_zeros)
    sampled_i, sampled_j, sampled_v = findnz(matX_train[usr_idx, itm_idx])
  end

  # Compute the objective function
  if model_type == "PoincareMF_random_negative" || model_type == "PoincareMF_softmax" || model_type == "PoincareMF_minibatch"
    objval = obj_val(matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, CS, lambda, usr_idx_len, itm_idx_len, usr_idx, itm_idx)
  elseif model_type == "sqdistPoincareMF_minibatch"
    gamma = 1.0
    objval = obj_val_sqdist(matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, CS, lambda, gamma, usr_idx_len, itm_idx_len, usr_idx, itm_idx)
  elseif model_type == "invPoincareMF_minibatch"
    objval = obj_val_inverse(matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, CS, lambda, usr_idx_len, itm_idx_len, usr_idx, itm_idx)
  elseif model_type == "PoincareMF_tfidf_minibatch"
    objval = obj_val_tfidf(matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, CS, lambda, usr_idx_len, itm_idx_len, usr_idx, itm_idx)
  else
    objval = 0
  end
  s = @sprintf "     Objective Function at step: %f" objval
  println(s)

  mt_theta = zeros(size(matTheta))
  nt_theta = zeros(size(matTheta))
  mt_beta = zeros(size(matBeta))
  nt_beta = zeros(size(matBeta))
  mt_biasU = zeros(size(vecBiasU))
  nt_biasU = zeros(size(vecBiasU))
  mt_biasI = zeros(size(vecBiasI))
  nt_biasI = zeros(size(vecBiasI))

  IsConverge = false
  itr = 0
  while IsConverge == false && itr < MaxItr
    itr += 1

    #
    # Sample data
    #
    if usr_batch_size != M && usr_batch_size != 0
      usr_idx, itm_idx, usr_idx_len, itm_idx_len = sample_data(M, N, usr_batch_size, matX_train, usr_zeros, itm_zeros)
      sampled_i, sampled_j, sampled_v = findnz(matX_train[usr_idx, itm_idx])
    end

    # lr = 0.1 * lr at the fist 10 epochs
    if itr <= 10
      lr_itr = lr / 10
    else
      lr_itr = lr
    end

    s = @sprintf "Step: %d  ---------------------------------- %d , %d , lr: %f" itr usr_idx_len itm_idx_len lr_itr
    print(s)

    #
    # Update (matBeta, vecBiasU) & (matTheta & vecBiasU)
    #
    update_matBeta_poincare(model_type, matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, CS, lambda, lr_itr, probDropout, usr_idx_len, itm_idx_len, usr_idx, itm_idx, itr, mu, mu2, mt_beta, nt_beta, mt_biasI, nt_biasI)
    update_matTheta_poincare(model_type, matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, CS, lambda, lr_itr, probDropout, usr_idx_len, itm_idx_len, usr_idx, itm_idx, itr, mu, mu2, mt_theta, nt_theta, mt_biasU, nt_biasU)

    #
    # Compute the objective function
    #
    if model_type == "PoincareMF_random_negative" || model_type == "PoincareMF_softmax" || model_type == "PoincareMF_minibatch"
      objval = obj_val(matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, CS, lambda, usr_idx_len, itm_idx_len, usr_idx, itm_idx)
    elseif model_type == "sqdistPoincareMF_minibatch"
      gamma = 1.0
      objval = obj_val_sqdist(matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, CS, lambda, gamma, usr_idx_len, itm_idx_len, usr_idx, itm_idx)
    elseif model_type == "invPoincareMF_minibatch"
      objval = obj_val_inverse(matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, CS, lambda, usr_idx_len, itm_idx_len, usr_idx, itm_idx)
    elseif model_type == "PoincareMF_tfidf_minibatch"
      objval = obj_val_tfidf(matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, CS, lambda, usr_idx_len, itm_idx_len, usr_idx, itm_idx)
    else
      objval = 0
    end
    train_loglikelihood[itr] = objval

    s = @sprintf "     Objective Function: %f" objval
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
        infer_N_eval_Pointcare(model_type, matX_train, matX_train, matTheta, vecBiasU, matBeta, vecBiasI, alpha, topK, vec_usr_idx, j, step_size, false)
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
      if model_type == "invPoincareMF_minibatch" || model_type == "sqdistPoincareMF_minibatch" || model_type == "PoincareMF_minibatch" || model_type == "PoincareMF_tfidf_minibatch"
        valid_precision[indx,:], valid_recall[indx,:] = evaluatePoincareMF(model_type, matX_valid, matX_train, matTheta, vecBiasU, matBeta, vecBiasI, topK, alpha)
      else
        valid_precision[indx,:], valid_recall[indx,:] = evaluateLogisticMF(matX_valid, matX_train, matTheta', vecBiasU, matBeta', vecBiasI, topK, 10., 1.)
      end
      println("validation precision: " * string(valid_precision[indx,:]))
      println("validation recall: " * string(valid_recall[indx,:]))

      #
      # Check whether the step performs the best. If yes, run testing and save model
      #
      if findmax(valid_precision[:,1])[2] == indx || (size(valid_precision,2)>1 && findmax(valid_precision[:,1])[2] != indx && findmax(valid_precision[:,2])[2] == indx)
        # Testing
        println("Testing ... ")
        indx = Int(itr / check_step)
        if model_type == "invPoincareMF_minibatch" || model_type == "sqdistPoincareMF_minibatch" || model_type == "PoincareMF_minibatch" || model_type == "PoincareMF_tfidf_minibatch"
          test_precision[indx,:], test_recall[indx,:] = evaluatePoincareMF(model_type, matX_test, matX_train, matTheta, vecBiasU, matBeta, vecBiasI, topK, alpha)
        else
          test_precision[indx,:], test_recall[indx,:] = evaluateLogisticMF(matX_test, matX_train, matTheta', vecBiasU, matBeta', vecBiasI, topK, 10., 1.)
        end
        println("testing precision: " * string(test_precision[indx,:]))
        println("testing recall: " * string(test_recall[indx,:]))

        # Save model
        file_name = string(model_type, "_K", K, "_", string(now())[1:10])
        write_model_PoincareMF(file_name, matTheta, matBeta, vecBiasU, vecBiasI, alpha, lr)

        println("BEST!!!!!")
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
      if model_type == "invPoincareMF_minibatch" || model_type == "sqdistPoincareMF_minibatch" || model_type == "PoincareMF_minibatch" || model_type == "PoincareMF_tfidf_minibatch"
        test_precision[indx,:], test_recall[indx,:] = evaluatePoincareMF(model_type, matX_test, matX_train, matTheta, vecBiasU, matBeta, vecBiasI, topK, alpha)
      else
        test_precision[indx,:], test_recall[indx,:] = evaluateLogisticMF(matX_test, matX_train, matTheta', vecBiasU, matBeta', vecBiasI, topK, 10., 1.)
      end
      println("testing precision: " * string(test_precision[indx,:]))
      println("testing recall: " * string(test_recall[indx,:]))
    end
  end

  return test_precision, test_recall, valid_precision, valid_recall, train_precision, train_recall, train_loglikelihood, bestTheta, bestBiasU, bestBeta, bestBiasI, matTheta, vecBiasU, matBeta, vecBiasI
end
