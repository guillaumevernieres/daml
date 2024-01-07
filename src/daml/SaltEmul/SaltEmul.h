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

#include "daml/Base/BaseEmul.h"
#include "SaltNet.h"

// -----------------------------------------------------------------------------

namespace daml {
  using ThreeDArray = std::vector<std::vector<std::vector<double>>>;
  // SaltEmul class derived from BaseEmul
  class SaltEmul : public BaseEmul<SaltNet> {
   public:
    // Constructor
    explicit SaltEmul(const std::string& infileName) : BaseEmul<SaltNet>(infileName) {}

    // -----------------------------------------------------------------------------
    // Override prepData
    std::tuple<torch::Tensor, torch::Tensor, std::vector<float>, std::vector<float>>
    prepData(const std::string& fileName, bool geoloc = false) override {

      // Read MOM6 restart
      netCDF::NcFile ncFile(fileName, netCDF::NcFile::read);

      // Read the dimensions
      int timeDim = ncFile.getDim("Time").getSize();
      int layerDim = ncFile.getDim("Layer").getSize();
      int lathDim = ncFile.getDim("lath").getSize();
      int lonhDim = ncFile.getDim("lonh").getSize();

      // Read the variables
      double temp[lonhDim][lathDim][layerDim];
      double salt[lonhDim][lathDim][layerDim];
      ncFile.getVar("Salt").getVar(salt);
      ncFile.getVar("Temp").getVar(temp);

      ncFile.close();
      std::cout << temp[0][0][0] << std::endl;

      // Store temp salt in a torch tensor
      int numPatterns(100);
      torch::Tensor patterns = torch::ones({numPatterns, 1, inputSize_}, torch::kFloat32);
      torch::Tensor targets = torch::ones({numPatterns, outputSize_}, torch::kFloat32);
      std::vector<float> lat;
      std::vector<float> lon;

      bool skipProfile = false;
      int cnt(0);
      for (size_t j = 0; j < lathDim; ++j) {
        for (size_t i = 0; i < lonhDim; ++i) {
          skipProfile = false;
          for (size_t z = 0; z < inputSize_; ++z) {
            if (salt[i][j][z] < 30.0 || temp[i][j][0]>20.0) {
              skipProfile = true;
            }
          }
          if (skipProfile) { continue; }
          for (size_t z = 0; z < inputSize_; ++z) {
            patterns[cnt][0][z] = temp[i][j][z];
            targets[cnt][z] = salt[i][j][z];
            if (not skipProfile) {
              std::cout << "z: " << z << std::endl;
              std::cout << "Temp: " << patterns[cnt][0][z].item<float>() << std::endl;
              std::cout << "Salt: " << targets[cnt][z].item<float>() <<std::endl;
              std::cout << "============================ " << cnt << std::endl;
            }
          }
          cnt +=1;
          if (cnt >= numPatterns) {
            std::cout << "--------------------------" << std::endl;
            goto exitLoops;
          }
        }
      }
    exitLoops:
      return std::make_tuple(patterns, targets, lon, lat);
    }

    // -----------------------------------------------------------------------------
    // Override predict
    void predict(const std::string& fileName, const std::string& fileNameResults) override {
      // Read the inputs/targets
      auto result = prepData(fileName, false);
      torch::Tensor inputs = std::get<0>(result);
      torch::Tensor targets = std::get<1>(result);

      std::cout << inputs.sizes() << std::endl;
      for (size_t j = 0; j < inputs.size(0); ++j) {
        for (size_t k = 0; k < inputSize_; ++k) {
          std::cout << "Temp: " << inputs[j][0][k].item<float>() << std::endl;
          std::cout << "Salt: " << targets[j][k].item<float>() << std::endl;
          std::cout << "============================" << j << std::endl;
        }
        torch::Tensor prediction = model_->forward(inputs[j]);
        std::cout << prediction.item<float>() << std::endl;
      }
    }
  };
}  // namespace daml
