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
  using array4D = std::vector<std::vector<std::vector<std::vector<float>>>>;

  // -----------------------------------------------------------------------------
  // Read variable from WOA file
  array4D readWOA(const std::string fileName, const std::string varName) {
    // Open the NetCDF file
    netCDF::NcFile ncFile(fileName, netCDF::NcFile::read);

    // Get the dimensions of the array
    int dim1 = ncFile.getDim("time").getSize();
    int dim2 = ncFile.getDim("lon").getSize();
    int dim3 = ncFile.getDim("lat").getSize();
    int dim4 = ncFile.getDim("depth").getSize();

    // Get the variable representing the 4D array
    std::vector<float> dataVector(dim1 * dim2 * dim3 * dim4);
    ncFile.getVar(varName).getVar(dataVector.data());

    // Reshape the vector into a 4D array
    array4D dataArray(dim1, std::vector<std::vector<std::vector<float>>>(
                          dim2, std::vector<std::vector<float>>(
                              dim3, std::vector<float>(
                                  dim4))));

    // Copy data from vector to 4D array
    size_t index = 0;
    for (size_t l = 0; l < dim4; ++l) {
      for (size_t k = 0; k < dim3; ++k) {
        for (size_t j = 0; j < dim2; ++j) {
          for (size_t i = 0; i < dim1; ++i) {
            dataArray[i][j][k][l] = dataVector[index++];
          }
        }
      }
    }
    return dataArray;
  }

  // -----------------------------------------------------------------------------
  // Read variable from WOA file
  bool checkprofile(const array4D &temp,
                    const array4D &salt,
                    const int i, const int j) {
    bool skipProfile;
    skipProfile = false;
    for (size_t jj = j-1; jj < j+1; ++jj) {
      for (size_t ii = i-1; ii < i+1; ++ii) {
        skipProfile = false;
        if (salt[0][ii][jj][0] > 50.0 || temp[0][ii][jj][0] > 50.0) {
          return true;
        }
        for (size_t z = 0; z < temp[0][0][0].size(); ++z) {
          if (salt[0][ii][jj][z] < 10.0 || temp[0][ii][jj][0] < 15.0) {
            return true;
          }
          if (salt[0][ii][jj][z] > 50.0 || temp[0][ii][jj][z] > 50.0) {
            return true;
          }
        }
      }
    }
    return skipProfile;
  }

  // -----------------------------------------------------------------------------
  // SaltEmul class derived from BaseEmul
  class SaltEmul : public BaseEmul<SaltNet> {
   public:
    // Constructor
    explicit SaltEmul(const std::string& infileName) : BaseEmul<SaltNet>(infileName) {}

    // -----------------------------------------------------------------------------
    // Override prepData
    std::tuple<torch::Tensor, torch::Tensor, std::vector<float>, std::vector<float>>
    prepData(const std::string& fileName, bool geoloc = false) override {
      // Read temp and salt from woa file
      array4D temp = readWOA(fileName, "t_an");
      array4D salt = readWOA(fileName, "s_an");

      // Prepare patterns/targets pairs
      // Input patterns: Temp, Salt and delta Temp
      // Output: delta Salt
      int numPatterns(batchSize_);
      torch::Tensor patterns = torch::ones({numPatterns, 3, inputSize_}, torch::kFloat32);
      torch::Tensor targets = torch::ones({numPatterns, outputSize_}, torch::kFloat32);
      std::vector<float> lat;
      std::vector<float> lon;

      std::vector<int> depthIndices = {0, 2, 4, 6, 8, 10, 12, 14, 16, 20,
        22, 24, 26, 28, 30, 40, 50, 60, 70, 100};


      int cnt(0);
      for (size_t j = 1; j < temp[0][0].size()-1; ++j) {
        for (size_t i = 1; i < temp[0][0][0].size()-1; ++i) {
          // check if the profile and its neighbor are valid
          bool skipProfile = checkprofile(temp, salt, i, j);
          if (skipProfile) { continue; }

          // construct patterns/targets pair
          int z = 0;
          for (int zindex : depthIndices) {
            std::cout << "z: " << z << std::endl;
            //for (size_t z = 0; z < inputSize_; ++z) {
            patterns[cnt][0][z] = temp[0][i][j][zindex];
            patterns[cnt][1][z] = salt[0][i][j][zindex];
            patterns[cnt][2][z] = temp[0][i-1][j][zindex] - temp[0][i][j][zindex];
            targets[cnt][z] = salt[0][i-1][j][zindex] - salt[0][i][j][zindex];

            std::cout << " woaz: " << zindex << std::endl;
            std::cout << "Temp prior: " << patterns[cnt][0][z].item<float>() << std::endl;
            std::cout << "Salt prior: " << patterns[cnt][1][z].item<float>() << std::endl;
            std::cout << "dT: " << patterns[cnt][2][z].item<float>() << std::endl;
            std::cout << "dS: " << targets[cnt][z].item<float>() <<std::endl;
            std::cout << "============================ " << cnt << std::endl;

            z += 1;
          }
          cnt +=1;
          if (cnt >= numPatterns) {
            std::cout << "--------------------------" << std::endl;
            return std::make_tuple(patterns, targets, lon, lat);
          }
        }
      }
      return std::make_tuple(patterns, targets, lon, lat);
    }

    // -----------------------------------------------------------------------------
    // Override predict
    void predict(const std::string& fileName, const std::string& fileNameResults) override {
      // Read the inputs/targets
      auto result = prepData(fileName, false);
      torch::Tensor inputs = std::get<0>(result);
      torch::Tensor targets = std::get<1>(result);

      torch::Tensor prediction = model_->forward(inputs);
      std::cout << "============================" << std::endl;
      std::cout << "======= PREDICTION =========" << std::endl;
      std::cout << "============================" << std::endl;
      for (size_t j = 0; j < inputs.size(0); ++j) {
        std::cout << "============================" << j << std::endl;
        for (size_t k = 0; k < inputSize_; ++k) {
          std::cout << "Temp: " << inputs[j][0][k].item<float>() << std::endl;
          std::cout << "Salt: " << inputs[j][1][k].item<float>() << std::endl;
          std::cout << "dT: " << inputs[j][2][k].item<float>() << std::endl;
          std::cout << "dS pred: " << prediction[j][k].item<float>() << std::endl;
          std::cout << "dS truth: " << targets[j][k].item<float>() << std::endl;
          std::cout << "Error: " << std::abs(prediction[j][k].item<float>() -
                                             targets[j][k].item<float>())<< std::endl;
          std::cout << "-------------------" << std::endl;
        }
      }
    }
  };
}  // namespace daml
