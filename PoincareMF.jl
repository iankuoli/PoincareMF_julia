include("evaluate.jl")
include("sample_data.jl")
include("PoincareMF_ParamUpdate.jl")
include("model_io.jl")


function PoincareMF(model_type::String, K::Int64, M::Int64, N::Int64,
                    matX_train::SparseMatrixCSC{Float64,Int64}, matX_test::SparseMatrixCSC{Float64,Int64}, matX_valid::SparseMatrixCSC{Float64,Int64},
                    ini_scale::Float64=0.003, alpha::Float64=1.0, lambda::Float64=0.0, lr::Float64=0.01, usr_batch_size::Int64=0, MaxItr::Int64=100,
                    topK::Array{Int64,1} = [10], test_step::Int64=0, check_step::Int64=5)
  #
  # MF based on Poincare distance
  # next work.
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
  vecBiasU = ini_scale * rand(M)
  vecBiasU[usr_zeros] = 0
  vecBiasI = ini_scale * rand(N)
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
    println("Update matBeta...")
    update_matBeta_poincare2(matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, lambda, usr_idx_len, itm_idx_len, usr_idx, itm_idx)
    #println(matBeta[1,:])
    println("Update matTheta...")
    update_matTheta_poincare2(matTheta, matBeta, vecBiasU, vecBiasI, matX_train, alpha, lambda, usr_idx_len, itm_idx_len, usr_idx, itm_idx)
    #println(matTheta[1,:])
    #
    # Validation
    #
    if mod(itr, check_step) == 0 && check_step > 0
      println("Validation ... ")
      indx = Int(itr / check_step)
      valid_precision[indx,:], valid_recall[indx,:] = evaluatePoincareMF(matX_valid, matX_train, matTheta, vecBiasU, matBeta, vecBiasI, topK, alpha)
      println("validation precision: " * string(valid_precision[indx,:]))
      println("validation recall: " * string(valid_recall[indx,:]))

      #
      # Check whether the step performs the best. If yes, run testing and save model
      #
      if findmax(valid_precision[:,1])[2] == Int(itr / check_step)
        # Testing
        println("Testing ... ")
        indx = Int(itr / check_step)
        test_precision[indx,:], test_recall[indx,:] = evaluatePoincareMF(matX_test, matX_train, matTheta, vecBiasU, matBeta, vecBiasI, topK, alpha)
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
      test_precision[indx,:], test_recall[indx,:] = evaluatePoincareMF(matX_test, matX_train, matTheta, vecBiasU, matBeta, vecBiasI, topK, alpha)
      println("testing precision: " * string(test_precision[indx,:]))
      println("testing recall: " * string(test_recall[indx,:]))
    end
  end

  return test_precision, test_recall, valid_precision, valid_recall, matTheta, vecBiasU, matBeta, vecBiasI
end
