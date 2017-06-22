include("LoadData.jl")
include("model_io.jl")
include("conf.jl")
include("evaluate.jl")

#
# Setting.
#
file_name = "PoincareMF_K2_2017-06-23"

#dataset = "MovieLens1M"
dataset = "MovieLens100K"
#dataset = "Lastfm1K"
#dataset = "Lastfm2K"
#dataset = "SmallToy"
env = 2
model_type = "PairPRPF"
topK = [5, 10, 15, 20]


#
# Load files to construct matrices.
#
training_path, testing_path, validation_path = train_filepath(dataset, env)
matX_train, matX_test, matX_valid, M, N = LoadUtilities(training_path, testing_path, validation_path)


#
# Load files to reconstruct the model.
#
matTheta, matBeta, vecGamma, vecDelta, alpha, lr = read_model_PoincareMF(file_name)


#
# Evaluate the performace of the model.
#
test_precision, test_recall, Tlog_likelihood = evaluate(matX_test, matX_train, matTheta, matBeta, topK, C, alpha)




matTheta

norm_theta = sqrt(diag(matTheta * matTheta'))
norm_beta = sqrt(diag(matBeta * matBeta'))








matTheta[6:10,:]





matBeta[1:5,:]





#
