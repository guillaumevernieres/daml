#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "eckit/config/YAMLConfiguration.h"
#include "eckit/filesystem/PathName.h"

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

      // Load model if asked in the config
      if (config_.has("ffnn.load model")) {

        std::string modelFileName;
        config_.get("ffnn.load model", modelFileName);
        torch::load(model_, modelFileName);
        model_->loadNorm(modelFileName);
        std::cout << "-----" << model_->inputMean << std::endl;
        for (const auto& pair : model_->named_parameters()) {
          std::cout << "Parameter name: " << pair.key()
                    << ", Size: " << pair.value().sizes() << std::endl;
        }
        for (const auto& pair : model_->named_buffers()) {
          std::cout << "Buffer name: " << pair.key()
                    << ", Size: " << pair.value().sizes() << std::endl;
          std::cout << "       values: " << pair.value() << std::endl;
        }
      }

      // Number of degrees of freedom in the FFNN
      info();
    }

    // Training
    void train(const torch::Tensor input, const torch::Tensor target) {
      // Loss function and optimizer.
      oops::Log::info() << "Define loss function and optimizer " << std::endl;
      torch::nn::MSELoss lossFn;
      torch::optim::Adam optimizer(model_->parameters(), torch::optim::AdamOptions(1e-3));

      std::cout << "nomalization in train:" << model_->inputMean << std::endl;
      // Train the model
      oops::Log::info() << "Train ..." << std::endl;
      for (size_t epoch = 0; epoch < epochs_; ++epoch) {
        // Forward pass.
        auto output = model_->forward(input);

        // Compute the loss.
        torch::Tensor loss = lossFn(output.view({-1}), target.view({-1}));
        if (epoch % 100 == 0) {
          updateProgressBar(epoch, epochs_, loss.item<float>());

          // Save the model
          torch::save(model_, modelOutputFileName_);
        }

        // Backward pass and optimization step.
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
      }
      // Save the normalization
      // TODO: it should be saved as part of the model, but for some reason it is not.
      //       figure out why ...
      model_->saveNorm(modelOutputFileName_);
      std::cout << std::endl;
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

    void info() {
      int numParams = 0;
      for (const auto& parameter : model_->parameters()) {
        numParams += parameter.numel();
      }

      std::cout << "Number of parameters: " << numParams << std::endl;
    }

  };
}  // namespace daml
