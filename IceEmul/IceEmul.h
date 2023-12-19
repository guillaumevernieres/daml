#pragma once

#include <netcdf>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "eckit/config/YAMLConfiguration.h"
#include "eckit/filesystem/PathName.h"

#include "nlohmann/json.hpp"
#include "oops/util/Logger.h"
#include "torch/torch.h"

#include "IceNet.h"

std::vector<float> readCice(const std::string fileName, const std::string varName) {
  // Open CICE restart file for reading
  netCDF::NcFile ncFile(fileName, netCDF::NcFile::read);

  // Get dime
  int dimLon = ncFile.getDim("ni").getSize();
  int dimLat = ncFile.getDim("nj").getSize();

  // Read the CICE variable
  std::vector<float> iceField(dimLat * dimLon);
  ncFile.getVar(varName).getVar(iceField.data());

  return iceField;
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
    std::cout << "] Loss: " << loss << std::setw(3) << static_cast<int>(percentage * 100) << "%\r";
    std::cout.flush();
}

class IceEmul {
 public:
  int inputSize_;
  int outputSize_;
  int hiddenSize_;
  size_t epochs_;
  std::string modelOutputFileName_;
  std::shared_ptr<IceNet> model_;

  // Constructor
  explicit IceEmul(const std::string infileName) :
    inputSize_(getSize(infileName, "ffnn.inputSize")),
    outputSize_(getSize(infileName, "ffnn.outputSize")),
    hiddenSize_(getSize(infileName, "ffnn.hiddenputSize")) {
    // Parse the configuration
    eckit::PathName infilePathName = infileName;
    eckit::YAMLConfiguration config(infilePathName);

    // Get the basic design parameters of the ffnn from the configuration.
    config.get("ffnn.outputSize", outputSize_);
    config.get("ffnn.hiddenSize", hiddenSize_);
    oops::Log::info() << "FFNN with " << inputSize_ << " inputs, "
                      << outputSize_ << " outputs" << std::endl;

    // Optimization parameters
    if (config.has("training")) {
      config.get("training.epochs", epochs_);
      config.get("training.model output", modelOutputFileName_);
    }

    // Initialize the FFNN
    model_ = std::make_shared<IceNet>(inputSize_, hiddenSize_, outputSize_);

    // Load model if asked in the config
    if (config.has("ffnn.load model")) {
      std::string modelFileName;
      config.get("ffnn.load model", modelFileName);
      torch::load(model_, modelFileName);
    }

    // Number of degrees of freedom in the FFNN
    size_t total_params = 0;
    for (const auto& parameter : model_->parameters()) {
        total_params += parameter.numel();
    }
    oops::Log::info() << "Number of parameters: " << total_params << std::endl;
  }

  // Training
  void train(const torch::Tensor input, const torch::Tensor target) {
    // Loss function and optimizer.
    oops::Log::info() << "Define loss fction and optimizer " << std::endl;
    torch::nn::MSELoss lossFn;
    torch::optim::Adam optimizer(model_->parameters(), torch::optim::AdamOptions(1e-3));

    // Train the model
    oops::Log::info() << "Train ..." << std::endl;
    for (size_t epoch = 0; epoch < epochs_; ++epoch) {
      // Forward pass.
      auto output = model_->forward(input);

      // Compute the loss.
      auto loss = lossFn(output.view({-1}), target);

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
    std::cout << std::endl;
  }

  // Prepare the inputs/targets
  std::tuple<torch::Tensor, torch::Tensor,
             std::vector<float>, std::vector<float>> prepData(std::string fileName,
                                                              bool geoloc = false) {
    // Read the patterns/targets    std::vector<float> lat = readCice(fileName, "ULAT");
    std::vector<float> lon = readCice(fileName, "ULON");
    std::vector<float> aice = readCice(fileName, "aice_h");
    std::vector<float> tsfc = readCice(fileName, "Tsfc_h");
    std::vector<float> sst = readCice(fileName, "sst_h");
    std::vector<float> sss = readCice(fileName, "sss_h");
    std::vector<float> sice = readCice(fileName, "sice_h");
    std::vector<float> hi = readCice(fileName, "hi_h");
    std::vector<float> hs = readCice(fileName, "hs_h");
    std::vector<float> mask = readCice(fileName, "umask");
    std::vector<float> tair = readCice(fileName, "Tair_h");

    int numPatterns(0);
    for (size_t i = 0; i < lat.size(); ++i) {
      if (mask[i] == 1 && lat[i] > 40.0 && aice[i] > 0.0 && aice[i] <= 1.0) {
        numPatterns+=1;
      }
    }
    std::cout << "Number of patterns: " << numPatterns << std::endl;

    torch::Tensor patterns = torch::empty({numPatterns, inputSize_}, torch::kFloat32);
    torch::Tensor targets = torch::empty({numPatterns}, torch::kFloat32);
    std::vector<float> lat_out;
    std::vector<float> lon_out;
    int cnt(0);
    for (size_t i = 0; i < lat.size(); ++i) {
      if (mask[i] == 1 && lat[i] > 40.0 && aice[i] > 0.0 && aice[i] <= 1.0) {
        patterns[cnt][0] = tair[i];
        patterns[cnt][1] = tsfc[i];
        patterns[cnt][2] = sst[i];
        patterns[cnt][3] = sss[i];
        patterns[cnt][4] = hs[i];
        patterns[cnt][5] = hi[i];
        patterns[cnt][6] = sice[i];

        targets[cnt] = aice[i];
        lat_out.push_back(lat[i]);
        lon_out.push_back(lon[i]);
        cnt+=1;
      }
    }
    return std::make_tuple(patterns, targets, lon_out, lat_out);
  }

  void predict(std::string fileName, std::string fileNameResults) {
    // Read the inputs/targets
    auto result = prepData(fileName, true);
    torch::Tensor inputs = std::get<0>(result);
    torch::Tensor targets = std::get<1>(result);
    std::vector lon = std::get<2>(result);
    std::vector lat = std::get<3>(result);

    // Loop through the patterns and predict
    torch::Tensor input = torch::ones({inputSize_});
    std::vector<float> ice_original;
    std::vector<float> ice_ffnn;
    // TODO(G): Store the jacobian in a 2D array
    std::vector<float> dcdt;
    std::vector<float> dcds;
    std::vector<float> dcdtsfc;
    std::vector<float> dcdtair;
    std::vector<float> dcdhi;
    std::vector<float> dcdhs;
    std::vector<float> dcdsi;
    for (size_t j = 0; j < targets.size(0); ++j) {
      for (size_t i = 0; i < inputSize_; ++i) {
        input[i] = inputs[j][i];
      }

      // Run the input through the FFNN
      torch::Tensor prediction = model_->forward(input);

      // Store results
      ice_original.push_back(targets[j].item<float>());
      ice_ffnn.push_back(prediction.item<float>());

      // Compute the Jacobian
      torch::Tensor doutdx = model_->jac(input);

      // Save the Jacobian elements into individual arrays
      // TODO(G): Store the jacobian in a 2D array
      dcdtair.push_back(doutdx[0].item<float>());
      dcdtsfc.push_back(doutdx[1].item<float>());
      dcdt.push_back(doutdx[2].item<float>());
      dcds.push_back(doutdx[3].item<float>());
      dcdhs.push_back(doutdx[4].item<float>());
      dcdhi.push_back(doutdx[5].item<float>());
      dcdsi.push_back(doutdx[6].item<float>());
    }

    // Save the prediction and Jacobian
    // TODO(G): Move into a separate function
    netCDF::NcFile ncFile(fileNameResults, netCDF::NcFile::replace);
    netCDF::NcDim dim = ncFile.addDim("n", ice_original.size());
    netCDF::NcDim dim2 = ncFile.addDim("n_inputs", inputSize_);

    ncFile.addVar("lon", netCDF::ncFloat, dim).putVar(lon.data());
    ncFile.addVar("lat", netCDF::ncFloat, dim).putVar(lat.data());
    ncFile.addVar("aice", netCDF::ncFloat, dim).putVar(ice_original.data());
    ncFile.addVar("aice_ffnn", netCDF::ncFloat, dim).putVar(ice_ffnn.data());
    ncFile.addVar("dcdt", netCDF::ncFloat, {dim}).putVar(dcdt.data());
    ncFile.addVar("dcds", netCDF::ncFloat, {dim}).putVar(dcds.data());
    ncFile.addVar("dcdhs", netCDF::ncFloat, {dim}).putVar(dcdhs.data());
    ncFile.addVar("dcdhi", netCDF::ncFloat, {dim}).putVar(dcdhi.data());
    ncFile.addVar("dcdsi", netCDF::ncFloat, {dim}).putVar(dcdsi.data());
    ncFile.addVar("dcdtsfc", netCDF::ncFloat, {dim}).putVar(dcdtsfc.data());
    ncFile.addVar("dcdtair", netCDF::ncFloat, {dim}).putVar(dcdtair.data());
  }

  // Initializers
  int getSize(const std::string infileName, const std::string paramName) {
    eckit::PathName infilePathName = infileName;
    eckit::YAMLConfiguration config(infilePathName);
    int param;
    config.get(paramName, param);
    return param;
  }
};
