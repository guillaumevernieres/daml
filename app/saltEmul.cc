#include "oops/mpi/mpi.h"
#include "oops/runs/Application.h"
#include "oops/runs/Run.h"
#include "oops/util/Logger.h"

#include "daml/SaltEmul/SaltEmul.h"

namespace daml {
  class SaltEmulApp : public oops::Application {
  public:
    explicit SaltEmulApp(const eckit::mpi::Comm & comm = oops::mpi::world())
      : Application(comm) {}

    // -----------------------------------------------------------------------------
    static const std::string classname() {return "daml::SaltEmulApp";}

    // -----------------------------------------------------------------------------
    int execute(const eckit::Configuration & config, bool /*validate*/) const {
      oops::Log::info() << "Initialize the FFNN" << std::endl;
      daml::SaltEmul saltEmul(config, getComm());

      // Generate patterns-targets pairs and train
      if (config.has("training")) {
        oops::Log::info() << "Prepare patterns/targets pairs" << std::endl;
        std::string fileName;
        config.get("training.ts profiles", fileName);
        auto result = saltEmul.prepData(fileName);
        torch::Tensor inputs = std::get<0>(result);
        torch::Tensor targets = std::get<1>(result);

        oops::Log::info() << "Train the FFNN" << std::endl;
        saltEmul.train(inputs, targets);
      }

      // Predictions
      if (config.has("prediction")) {
        oops::Log::info() << "Predict" << std::endl;
        std::string fileName;
        std::string fileNameResults;
        config.get("prediction.output filename", fileNameResults);
        config.get("prediction.ts profiles", fileName);
        int batchSize;
        config.get("prediction.batch size", batchSize);
        saltEmul.predict(fileName, fileNameResults, batchSize);
      }
      return 0;
    }
    // -----------------------------------------------------------------------------
  private:
    std::string appname() const {
      return "daml::saltEmul";
    }
    // -----------------------------------------------------------------------------

  };
}

int main(int argc, char* argv[]) {
  oops::Run run(argc, argv);
  daml::SaltEmulApp saltemulapp;
  return run.execute(saltemulapp);
}
