addprocs(7)

include("LoadData.jl")
@everywhere include("PoincareMF.jl")
include("conf.jl")



#
# Setting.
#
#dataset = "MovieLens1M"
#dataset = "MovieLens100K"
#dataset = "Lastfm1K"
#dataset = "Lastfm2K"
#dataset = "Lastfm360K"
#dataset = "SmallToy"
dataset = "SmallToy2"
#dataset = "SmallToy3"

env = 2
model_type = "PoincareMF"
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
usr_batch_size = 0
check_step = 5
test_step = 5

if dataset == "SmallToy" || dataset == "SmallToy2"
  topK = [1, 2, 3, 5]
  Ks = [2]
elseif dataset == "SmallToy3"
  topK = [1]
  Ks = [2]
else
  topK = [5, 10, 15, 20]
  Ks = [100]
end
listBestPrecisionNRecall = zeros(length(Ks), length(topK)*2)
CS = 2.0

#model_type = "PoincareMF_random_negative"
#model_type = "PoincareMF_minibatch"
model_type = "sqdistPoincareMF_minibatch"
#model_type = "invPoincareMF_minibatch"
#model_type = "PoincareMF_tfidf_minibatch"
#model_type = "PoincareMF_softmax"
#model_type = "PoincareMF_log_softmax"
#model_type = "PoincareMF_softmax"

if model_type == "PoincareMF_random_negative"
  lr = 0.005
  MaxItr = 120
  alpha = 1.
  lambda = 0.0
elseif model_type == "PoincareMF_minibatch"
  MaxItr = 800
  alpha = 1.
  if CS == 2.0
    lr = 0.001
    mu =0.3
    mu2 = 0.5
    lambda = 0.5
  elseif CS == 1.0
    lr = 0.001
    mu = 0.3
    mu2 = 0.5
    lambda = 0.1
  end

elseif model_type == "sqdistPoincareMF_minibatch"
  MaxItr = 800
  alpha = 1.
  lr = 0.001
  mu =0.3
  mu2 = 0.5
  lambda = 0.

elseif model_type == "PoincareMF_tfidf_minibatch"
  MaxItr = 2000
  alpha = 1.
  if CS == 2.0
    lr = 0.001
    mu =0.3
    mu2 = 0.5
    lambda = 0.5
  elseif CS == 1.0
    lr = 0.001
    mu = 0.4
    mu2 = 0.6
    lambda = 0.1
  end

elseif model_type == "PoincareMF_softmax"
  lr = 0.001
  MaxItr = 300
  alpha = 1.
  lambda = 0.0

elseif model_type == "PoincareMF_log_softmax"
  lr = 0.1
  MaxItr = 300
  alpha = 1.
  lambda = 0.0

elseif model_type == "PoincareMF_log_softmax"
  lr = 0.001
  MaxItr = 300
  alpha = 1.
  lambda = 0.0

end

matTheta = zeros(M, 1)
matBeta = zeros(N, 1)
vecBiasU = zeros(M)
vecBiasI = zeros(N)
bestTheta = zeros(M, 1)
bestBeta = zeros(N, 1)
bestBiasU = zeros(M)
bestBiasI = zeros(N)
test_precision = 0
test_recall = 0
valid_precision = 0
valid_recall = 0
train_loglikelihood = 0
train_precision = 0
train_recall = 0
Ini = true
probDropout = 0.
for k = 1:length(Ks)
  K = Ks[k]
  ini_scale = 0.01 / K
  test_precision, test_recall,
  valid_precision, valid_recall,
  train_precision, train_recall, train_loglikelihood,
  bestTheta, bestBiasU, bestBeta, bestBiasI,
  matTheta, vecBiasU, matBeta, vecBiasI = PoincareMF(Ini, probDropout, model_type, K, M, N,
                                                     matTheta, matBeta, vecBiasU, vecBiasI,
                                                     matX_train, matX_test, matX_valid,
                                                     ini_scale, alpha, CS, lambda, lr, mu, mu2,
                                                     usr_batch_size, MaxItr, topK, test_step, check_step)

  (bestVal, bestIdx) = findmax(test_precision[:,1])
  listBestPrecisionNRecall[k,:] = [test_precision[bestIdx, :]; test_recall[bestIdx, :]]

  open(results_path, "a") do f
    writedlm(f, listBestPrecisionNRecall[k,:]')
  end
end
writedlm(results_path, listBestPrecisionNRecall)

listBestPrecisionNRecall
size(matTheta)

using Plots
Plots.plot(matTheta[:,1:40]')
Plots.plot(bestTheta[:,1:40]')
Plots.plot(matBeta[:,1:40]')
Plots.plot(vecBiasU[1:40])
Plots.plot(train_precision)
Plots.plot(train_loglikelihood)
Plots.plot(test_precision)
Plots.surface(inference_Poincare2(collect(1:size(matTheta,2)), matTheta, matBeta, vecBiasU, vecBiasI))
Plots.surface(inference_Poincare2(collect(1:size(matTheta,2)), bestTheta, bestBeta, bestBiasU, bestBiasI))
Plots.surface(sqrt.(matX_train))

println(matTheta[1,:])
println(matTheta[2,:])
println(matBeta[1,:])
println(matBeta[2,:])
test_precision

QQ = inference_Poincare2(collect(1:size(matTheta,2)), matTheta, matBeta, vecBiasU, vecBiasI)


open("model_last.csv", "w") do f
  writedlm(f, matTheta[1,:]')
  writedlm(f, matTheta[2,:]')
  writedlm(f, matBeta[1,:]')
  writedlm(f, matBeta[2,:]')
end

open("model_best.csv", "w") do f
  writedlm(f, bestTheta[1,:]')
  writedlm(f, bestTheta[2,:]')
  writedlm(f, bestBeta[1,:]')
  writedlm(f, bestBeta[2,:]')
end


ttt[25:30]


matTheta[:,20:25]'



XXX = broadcast(/, matX_train, ttt)
Plots.plot(ttt')





#
