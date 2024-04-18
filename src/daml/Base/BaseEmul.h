#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <mpi.h>
#include <string>
#include <tuple>
#include <vector>

#include "eckit/config/YAMLConfiguration.h"
#include "eckit/filesystem/PathName.h"
#include "eckit/mpi/Comm.h"

#include "nlohmann/json.hpp"
#include "oops/mpi/mpi.h"
#include "oops/util/Logger.h"
#include "torch/torch.h"

// -----------------------------------------------------------------------------

namespace daml {

  // -----------------------------------------------------------------------------
  /// Utilities
  // -----------------------------------------------------------------------------
  // -----------------------------------------------------------------------------
  /// Emulator base class
  // -----------------------------------------------------------------------------
  template <typename Net>
  class BaseEmul {
   public:
    int inputSize_;
    int outputSize_;
    int hiddenSize_;
    int kernelSize_;
    int stride_;
    int batchSize_;
    size_t epochs_;
    std::string modelOutputFileName_;
    std::shared_ptr<Net> model_;
    //eckit::PathName configFile_;
    const eckit::mpi::Comm & comm_;
    const eckit::Configuration & config_;

   public:
    // Constructor
    explicit BaseEmul(const eckit::Configuration & config, const eckit::mpi::Comm & comm):
      comm_(comm), config_(config),
      inputSize_(getSize(config, "ffnn.inputSize")),
      outputSize_(getSize(config, "ffnn.outputSize")),
      hiddenSize_(getSize(config, "ffnn.hiddenSize")) {
      // Check pyTorch version
      std::cout << "PyTorch Version: "
              << TORCH_VERSION_MAJOR << "."
              << TORCH_VERSION_MINOR << "."
              << TORCH_VERSION_PATCH << std::endl;

      // Check thread info
      unsigned int maxThreads = std::thread::hardware_concurrency();
      oops::Log::info() << "Maximum threads supported: " << maxThreads << std::endl;

      // Get the basic design parameters of the ffnn from the configuration.
      oops::Log::info() << "FFNN with " << inputSize_ << " inputs, "
                        << outputSize_ << " outputs" << std::endl;

      // Get the parameters for the convolution layer
      if (config_.has("ffnn.kernelSize")) {
          config_.get("ffnn.kernelSize", kernelSize_);
          config_.get("ffnn.stride", stride_);
      }

      // Optimization parameters
      if (config_.has("training")) {
        config_.get("training.epochs", epochs_);
        config_.get("training.model output", modelOutputFileName_);
        config_.get("training.batch size", batchSize_);
      }

      // Initialize the FFNN
      model_ = std::make_shared<Net>(inputSize_, hiddenSize_, outputSize_, kernelSize_, stride_);
      model_->initWeights();

      // Load model if asked in the config
      if (config_.has("ffnn.load model")) {

        std::string modelFileName;
        config_.get("ffnn.load model", modelFileName);
        torch::load(model_, modelFileName);
        model_->loadNorm(modelFileName);
        /*
        std::cout << "----- mean: " << model_->inputMean << std::endl;
        std::cout << "----- std dev: " << model_->inputStd << std::endl;
        for (const auto& pair : model_->named_parameters()) {
          std::cout << "Parameter name: " << pair.key()
                    << ", Size: " << pair.value().sizes() << std::endl;
        }
        for (const auto& pair : model_->named_buffers()) {
          std::cout << "Buffer name: " << pair.key()
                    << ", Size: " << pair.value().sizes() << std::endl;
          std::cout << "       values: " << pair.value() << std::endl;
        }
        */
      }

      // Number of degrees of freedom in the FFNN
      info();
    }

    // Training
    void train(const torch::Tensor input, const torch::Tensor target) {
      // Loss function and optimizer.
      oops::Log::trace() << "Define loss function and optimizer " << std::endl;
      torch::nn::MSELoss lossFn;
      torch::optim::AdamOptions adamOptions(0.001);
      //adamOptions.betas(std::make_tuple(0.5, 0.5));
      //adamOptions.eps(1e-6);
      //adamOptions.weight_decay(0.0);
      //adamOptions.amsgrad(false);
      torch::optim::Adam optimizer(model_->parameters(),
                                   adamOptions);

      // MPI info
      int worldSize;
      MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      // Get info about the input/target distribution
      int localBatchSize = input.size(1);
      int totalBatchSize;
      MPI_Allreduce(&localBatchSize, &totalBatchSize,
                    1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      input.set_requires_grad(true);

      // Train the model
      float finalLoss(0.0);
      oops::Log::trace() << "Train ..." << std::endl;
      for (size_t epoch = 0; epoch < epochs_; ++epoch) {
        // Setup the model for training
        //model_->train();

        // Forward pass.
        auto output = model_->forward(input);

        // Compute the loss.
        torch::Tensor loss = lossFn(output.view({-1}), target.view({-1}));

        //std::cout << "**** prediction: " << output.view({-1}) << std::endl;
        //std::cout << "**** target: " << target.view({-1}) << std::endl;

        // // Compute loss from the norm of the Jacobian
        // torch::Tensor frobeniusNorm = torch::tensor(0.0, torch::requires_grad(true));
        // for (int i = 0; i < input.size(1); ++i) {
        //   auto one_hot = torch::zeros_like(input);
        //   one_hot.select(1, i) = 1.0;

        //   // Reset gradients
        //   model_->zero_grad();

        //   if (input.grad().defined()) {
        //     input.grad().detach_();
        //     input.grad().zero_();
        //   }

        //   // Backward pass for this input dimension
        //   output.backward(one_hot, true /* retain_graph */, true /* create_graph */);

        //   // Accumulate the square of gradients (approximation of Frobenius norm of Jacobian)
        //   frobeniusNorm = frobeniusNorm + input.grad().pow(2).sum();
        // }

        auto lambda = 0.000; // Regularization strength
        //torch::Tensor jacobianNorm = torch::tensor(0.0, torch::requires_grad(true));
        //jacobianNorm = model_->jacNorm(input);
        //auto totalLoss = loss;  // + lambda * frobeniusNorm;

        finalLoss = loss.item<float>();
        oops::Log::trace() << "Compute loss " << finalLoss << std::endl;

        //

        // Save the model
        if (epoch % 100 == 0) {
          updateProgressBar(epoch, epochs_, loss.item<float>());
          torch::save(model_, modelOutputFileName_);
        }

        // Backward pass
        oops::Log::trace() << "Backward pass " << std::endl;
        optimizer.zero_grad();
        //totalLoss.backward();
        loss.backward();

        // Scale gradients by the local batch size
        /*
        oops::Log::trace() << "gradient scaling" << std::endl;
        for (auto& param : model_->parameters()) {
          param.grad().data() *= static_cast<float>(localBatchSize);
        }
        comm_.barrier();
        */

        // Aggregate gradients
        oops::Log::trace() << "Aggregate gradient" << std::endl;
        for (auto& param : model_->parameters()) {
          if (param.grad().defined()) {
            MPI_Allreduce(MPI_IN_PLACE, param.grad().data_ptr(),
                          param.grad().numel(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            param.grad().data() /= static_cast<float>(totalBatchSize);
          }
        }
        comm_.barrier();

        // Gradient descent
        oops::Log::trace() << "Gradient descent" << std::endl;
        comm_.barrier();
        optimizer.step();
      }

      /*
      if (rank == 0) {
        oops::Log::info() << "Final loss: " << finalLoss << std::endl;
        oops::Log::info() << "normalization in train:" << model_->inputMean << std::endl;
        torch::save(model_, modelOutputFileName_);
        // Save the normalization
        // TODO: it should be saved as part of the model, but for some reason it is not.
        //       figure out why ...
        model_->saveNorm(modelOutputFileName_);
      }
      */
    }

    // Prepare patterns/targets pairs
    virtual std::tuple<torch::Tensor,
                       torch::Tensor,
                       std::vector<float>,
                       std::vector<float>,
                       torch::Tensor,
                       torch::Tensor>
                  prepData(const std::string& fileName, bool geoloc = false, int n = -999) = 0;

    // Forward propagation and Jacobian
    virtual void predict(const std::string& fileName,
                         const std::string& fileNameResults,
                         const int n) = 0;

    // Initializers
    int getSize(const eckit::Configuration & config, const std::string& paramName) {
      int param;
      config.get(paramName, param);
      return param;
    }

    void updateProgressBar(int progress, int total, float loss) {
      if (comm_.rank() == 0) {
        const int barWidth = 50;
        float percentage = static_cast<float>(progress) / total;
        int barLength = static_cast<int>(percentage * barWidth);
        std::cout << "[";
        for (int i = 0; i < barWidth; ++i) {
          if (i < barLength) {
            std::cout << "=";
          } else {
            std::cout << " ";
          }
        }
        std::cout << "] " << std::setw(3) << static_cast<int>(percentage * 100)
                  << "% "<< "Loss: " << loss << "\r";
        std::cout.flush();
      }
    }

    void info() {
      int numParams = 0;
      for (const auto& parameter : model_->parameters()) {
        numParams += parameter.numel();
      }

      std::cout << "Number of parameters: " << numParams << std::endl;
    }

  };
}  // namespace daml
