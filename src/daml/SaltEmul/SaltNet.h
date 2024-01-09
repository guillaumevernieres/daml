#pragma once

#include <utility>
#include <torch/torch.h>

#include "oops/util/Logger.h"

// Define the Feed Forward Neural Net model
struct SaltNet : torch::nn::Module {
  SaltNet(int inputSize, int hiddenSize, int outputSize,
          int kernelSize, int stride) {
    oops::Log::trace() << "Net: " << inputSize
                       << outputSize << hiddenSize << kernelSize << std::endl;

    // Define the convolution layers
    conv = register_module("conv",
                  torch::nn::Conv1d(torch::nn::Conv1dOptions(3,
                                                             hiddenSize,
                                                             kernelSize).stride(stride)));

    // Define the layers.
    int nin = hiddenSize * (inputSize - kernelSize + 1);
    fc1 = register_module("fc1", torch::nn::Linear(nin, hiddenSize));
    fc2 = register_module("fc2", torch::nn::Linear(hiddenSize, outputSize));
  }

  // Implement the forward pass
  torch::Tensor forward(torch::Tensor x) {
    x = conv(x);
    x = x.view({x.size(0), -1});
    x = torch::sigmoid(fc1(x));
    //x = fc1(x);
    torch::Tensor ds = fc2(x);
    return ds;
  }

  // Compute the Jacobian (dout/dx)
  torch::Tensor jac(torch::Tensor x) {
    auto xp = torch::from_blob(x.data_ptr(), {x.size(0)}, torch::requires_grad());
    auto y = this->forward(xp);
    y.backward();
    return xp.grad();
  }

  // Data members
  torch::nn::Conv1d conv{nullptr};
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
};
