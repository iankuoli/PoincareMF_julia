using HDF5

function write_model_PoincareMF(file_name, matTheta, matBeta, vecGamma, vecDelta, alpha, lr)
  path = string("model/", file_name, ".h5")

  if isfile(path)
    rm(path)
  end

  h5write(path, "PoincareMF/matTheta", matTheta)
  h5write(path, "PoincareMF/matBeta", matBeta)
  h5write(path, "PoincareMF/vecGamma", vecGamma)
  h5write(path, "PoincareMF/vecDelta", vecDelta)
  h5write(path, "PoincareMF/alpha", alpha)
  h5write(path, "PoincareMF/lr", lr)
end

function read_model_PoincareMF(file_name)
  path = string("model/", file_name, ".h5")
  matTheta = h5read(path, "PoincareMF/matTheta")
  matBeta = h5read(path, "PoincareMF/matBeta")
  vecGamma = h5read(path, "PoincareMF/vecGamma")
  vecDelta = h5read(path, "PoincareMF/vecDelta")
  alpha = h5read(path, "PoincareMF/alpha")
  lr = h5read(path, "PoincareMF/lr")

  return matTheta, matBeta, vecGamma, vecDelta, alpha, lr
end
