addprocs(2)

include("LoadData.jl")
@everywhere include("PoincareMF.jl")
include("conf.jl")



#
# Setting.
#
#dataset = "MovieLens1M"
#dataset = "MovieLens100K"
#dataset = "Lastfm1K"
dataset = "Lastfm2K"
#dataset = "Lastfm360K"
#dataset = "SmallToy"

env = 1
model_type = "PoincareMF"

#Ks = [5, 20, 50, 100, 150, 200]
Ks = [10]
topK = [5, 10, 15, 20]
usr_batch_size = 0
test_step = 2
check_step = 2
MaxItr = 40
alpha = 1.

results_path = joinpath("results", string(dataset, "_", model_type, "_alpha", Int(alpha), ".csv"))

#
# Initialize the hyper-parameters.
#
(prior, ini_scale, batch_size, MaxItr, test_step, check_step, lr, lambda, lambda_Theta, lambda_Beta, lambda_B) = train_setting(dataset, "PRPF")


#
# Load files to construct training set (utility matrices), validation set and test set.
#
training_path, testing_path, validation_path = train_filepath(dataset)
matX_train, matX_test, matX_valid, M, N = LoadUtilities(training_path, testing_path, validation_path)


#
# Training
#
lr = 0.000001
check_step = 2
test_step = 0
listBestPrecisionNRecall = zeros(length(Ks), length(topK)*2)
for k = 1:length(Ks)
  K = Ks[k]
  test_precision, test_recall, valid_precision, valid_recall, matTheta, vecGamma, matBeta, vecDelta = PoincareMF(model_type, K, M, N,
                                                                                                      matX_train, matX_test, matX_valid,
                                                                                                      ini_scale, alpha, lr, usr_batch_size, MaxItr,
                                                                                                      topK, test_step, check_step)
  end

  println(matTheta[1,:])

  (bestVal, bestIdx) = findmax(test_precision[:,1])
  listBestPrecisionNRecall[k,:] = [test_precision[bestIdx, :]; test_recall[bestIdx, :]]

  open(results_path, "a") do f
    writedlm(f, listBestPrecisionNRecall[k,:]')
  end
end
writedlm(results_path, listBestPrecisionNRecall)

listBestPrecisionNRecall















#
















#