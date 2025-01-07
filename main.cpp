#include <string>
#include <chrono>
#include <torch/torch.h>
#include "dataset.h"
#include "model.h"
#include "trainer.h"


// void initialize_weights(Model& model) {
//     for (auto& named_param : model.named_parameters()) {
//         if (named_param.key().find("weight") != std::string::npos) {
//             if (named_param.value().dim() == 4) {  // Conv2d weight
//                 torch::nn::init::xavier_uniform_(named_param.value());
//             } else if (named_param.value().dim() == 2) {  // Linear weight
//                 torch::nn::init::xavier_uniform_(named_param.value());
//             }
//         } else if (named_param.key().find("bias") != std::string::npos) {
//             torch::nn::init::constant_(named_param.value(), 0.0);
//         }
//     }
// }

// void initialize_weights(Model& model) {

// }

void initialize_weights(Model& model) {
    for (auto& named_param : model.named_parameters()) {
        if (named_param.key().find("weight") != std::string::npos) {
            // 如果是权重参数
            if (named_param.value().dim() == 4) {  // Conv2d weight
                torch::nn::init::kaiming_uniform_(named_param.value(), 0);
            } else if (named_param.value().dim() == 2) {  // Linear weight
                torch::nn::init::kaiming_uniform_(named_param.value(), 0);
            }
        } else if (named_param.key().find("bias") != std::string::npos) {
            // 如果是偏置参数
            torch::nn::init::constant_(named_param.value(), 0.0);
        }
    }
}

int main() {
  // 超参数设置
  std::string data_root = "./data";
  int train_batch_size = 32;
  int test_batch_size = 32;
  int total_epoch_num = 30;
  int log_interval = 10;
  int num_workers = 4;

  // 设置随机数种子
  torch::manual_seed(1);

  // 获取设备类型
  torch::DeviceType device_type = torch::kCPU;
  if (torch::cuda::is_available()) {
    device_type = torch::kCUDA;
  }
  torch::Device device(device_type);

  // 构造网络
  Model model;
  initialize_weights(model);
  model.to(device);

  // 设置优化器
  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

  // 构造训练和测试数据集
  Dataset train_dataset("./data/indice/test_indice.csv", "./data/training_dataset", Dataset::Phase::Train);
  Dataset test_dataset("./data/indice/test_indice.csv", "./data/training_dataset", Dataset::Phase::Test);
  auto train_start = std::chrono::high_resolution_clock::now();
  // Trainer初始化
  Trainer trainer(log_interval);
  for (size_t epoch = 1; epoch <= total_epoch_num; ++epoch) {
    // 运行训练
    trainer.train(
        epoch,
        model,
        optimizer,
        device,
        train_dataset,
        train_batch_size,
        num_workers);

    // 运行测试
    trainer.test(model, device, test_dataset, test_batch_size, num_workers);
  }
  auto train_end = std::chrono::high_resolution_clock::now();
  auto train_duration = std::chrono::duration_cast<std::chrono::microseconds>(train_end - train_start).count();
  std::cout << "Total training time: " << train_duration << " seconds" << std::endl;

  torch::serialize::OutputArchive archive;
	model.save(archive);
	archive.save_to("model.pt");
	printf("Save the training result to model.pt.\n");

  return 0;
}