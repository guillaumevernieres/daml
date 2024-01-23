#include "oops/mpi/mpi.h"
#include "oops/runs/Application.h"
#include "oops/runs/Run.h"
#include "oops/util/Logger.h"

#include "daml/IceEmul/IceEmul.h"

namespace daml {
  class IceEmulApp : public oops::Application {
  public:
    explicit IceEmulApp(const eckit::mpi::Comm & comm = oops::mpi::world())
      : Application(comm) {}

    // -----------------------------------------------------------------------------
    static const std::string classname() {return "daml::IceEmulApp";}

    // -----------------------------------------------------------------------------
    int execute(const eckit::Configuration & config, bool /*validate*/) const {
      oops::Log::info() << "Initialize the FFNN" << std::endl;
      daml::IceEmul iceEmul(config, getComm());

      // Generate patterns-targets pairs and train
      if (config.has("training")) {
        oops::Log::info() << "Prepare patterns/targets pairs" << std::endl;
        std::string fileName;
        config.get("training.cice history", fileName);
        auto result = iceEmul.prepData(fileName);
        torch::Tensor inputs = std::get<0>(result);
        torch::Tensor targets = std::get<1>(result);

        oops::Log::info() << "Initialize the normalization" << std::endl;
        torch::Tensor mean = std::get<4>(result);
        torch::Tensor std = std::get<5>(result);
        iceEmul.model_->initNorm(mean, std);

        oops::Log::info() << "Train the FFNN" << std::endl;
        iceEmul.train(inputs, targets);
      }

      // Predictions
      if (config.has("prediction")) {
        oops::Log::info() << "Predict" << std::endl;
        std::string fileName;
        config.get("prediction.cice history", fileName);
        std::string fileNameResults;
        config.get("prediction.output filename", fileNameResults);
        iceEmul.predict(fileName, fileNameResults);
      }
      return 0;
    }
    // -----------------------------------------------------------------------------
   private:
    std::string appname() const {
      return "gdasapp::iceEmul";
    }
    // -----------------------------------------------------------------------------

  };
}

int main(int argc, char* argv[]) {
  oops::Run run(argc, argv);
  daml::IceEmulApp iceemulapp;
  return run.execute(iceemulapp);
}
