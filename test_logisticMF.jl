addprocs(7)

include("LoadData.jl")
@everywhere include("LogisticMF.jl")
include("conf.jl")



#
# Setting.
#
#dataset = "MovieLens1M"
#dataset = "MovieLens100K"
#dataset = "Lastfm1K"
#dataset = "Lastfm2K"
#dataset = "Lastfm360K"
dataset = "SmallToy"
#dataset = "SmallToy2"

#
# Initialize the hyper-parameters.
#

ini_scale = 0.003
model_type = "LogisticMF"
env = 2
usr_batch_size = 0
check_step = 5
test_step = 5

if dataset == "SmallToy" || dataset == "SmallToy2"
  topK = [1, 2, 3, 5]
  Ks = [2]
else
  topK = [5, 10, 15, 20]
  Ks = [100]
  #Ks = [5, 20, 50, 100, 150, 200]
end

#
# Load files to construct training set (utility matrices), validation set and test set.
#
training_path, testing_path, validation_path = train_filepath(dataset)
matX_train, matX_test, matX_valid, M, N = LoadUtilities(training_path, testing_path, validation_path)


#
# Training
#
#lr = 0.001   # for SmallToy
#lr = 0.01   # for Last.fm2K
lr = 0.02   # for Last.fm2K
MaxItr = 1000
alpha = 1.
lambda = 0.0

results_path = joinpath("results", string(dataset, "_", model_type, "_alpha", Int(alpha), ".csv"))
listBestPrecisionNRecall = zeros(length(Ks), length(topK)*2)

matTheta = 0
matBeta = 0
vecBiasU = 0
vecBiasI = 0
test_precision = 0
test_recall = 0
train_precision = 0
train_recall = 0
mu = 0.3
mu2 = 0.5
for k = 1:length(Ks)
  K = Ks[k]
  ini_scale = 0.1 / K
  test_precision, test_recall,
  valid_precision, valid_recall,
  train_precision, train_recall,
  bestTheta, bestBiasU, bestBeta, bestBiasI,
  matTheta, vecBiasU, matBeta, vecBiasI = LogisticMF(model_type, K, M, N,
                                                     matX_train, matX_test, matX_valid,
                                                     ini_scale, alpha, lambda, lr, usr_batch_size, MaxItr,
                                                     topK, test_step, check_step, mu, mu2)

  (bestVal, bestIdx) = findmax(test_precision[:,1])
  listBestPrecisionNRecall[k,:] = [test_precision[bestIdx, :]; test_recall[bestIdx, :]]

  open(results_path, "a") do f
    writedlm(f, listBestPrecisionNRecall[k,:]')
  end
end
writedlm(results_path, listBestPrecisionNRecall)

listBestPrecisionNRecall

using Plots
Plots.plot(test_precision)
Plots.plot(matTheta)
Plots.plot(train_precision)
Plots.surface(inferenceLogisticMF(collect(1:50), matTheta, vecBiasU, matBeta, vecBiasI))
Plots.surface(inferenceLogisticMF(collect(1:50), bestTheta, bestBiasU, bestBeta, bestBiasI))
Plots.surface(matX_train)


println(matBeta[:,2]')

open("model.csv", "w") do f
  writedlm(f, matTheta[:,1]')
  writedlm(f, matTheta[:,2]')
  writedlm(f, matBeta[:,1]')
  writedlm(f, matBeta[:,2]')
end












matTheta[:,20:25]'










#
