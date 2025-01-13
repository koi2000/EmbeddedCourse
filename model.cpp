#include "model.h"

Model::Model() {
    // 初始化卷积层
    conv1 = register_module("conv1", torch::nn::Sequential(
        torch::nn::Conv1d(torch::nn::Conv1dOptions(1, 3, 6).stride(2).padding(2)),
        torch::nn::ReLU(true),
        torch::nn::BatchNorm1d(3)
    ));

    conv2 = register_module("conv2", torch::nn::Sequential(
        torch::nn::Conv1d(torch::nn::Conv1dOptions(3, 5, 5).stride(2).padding(2)),
        torch::nn::ReLU(true),
        torch::nn::BatchNorm1d(5)
    ));

    conv3 = register_module("conv3", torch::nn::Sequential(
        torch::nn::Conv1d(torch::nn::Conv1dOptions(5, 10, 4).stride(2).padding(2)),
        torch::nn::ReLU(true),
        torch::nn::BatchNorm1d(10)
    ));

    conv4 = register_module("conv4", torch::nn::Sequential(
        torch::nn::Conv1d(torch::nn::Conv1dOptions(10, 20, 4).stride(2).padding(2)),
        torch::nn::ReLU(true),
        torch::nn::BatchNorm1d(20)
    ));

    conv5 = register_module("conv5", torch::nn::Sequential(
        torch::nn::Conv1d(torch::nn::Conv1dOptions(20, 20, 4).stride(2).padding(2)),
        torch::nn::ReLU(true),
        torch::nn::BatchNorm1d(20)
    ));

    // 初始化自适应池化层
    adaptive_pool = register_module("adaptive_pool", torch::nn::AdaptiveAvgPool1d(1));

    // 初始化全连接层
    fc1 = register_module("fc1", torch::nn::Sequential(
        torch::nn::Dropout(0.5),
        torch::nn::Linear(20, 10) // 20 是通道数
    ));

    fc2 = register_module("fc2", torch::nn::Linear(10, 2)); // 10 是 fc1 的输出
}

torch::Tensor Model::forward(torch::Tensor input) {
    auto conv1_output = conv1->forward(input);
    auto conv2_output = conv2->forward(conv1_output);
    auto conv3_output = conv3->forward(conv2_output);
    auto conv4_output = conv4->forward(conv3_output);
    auto conv5_output = conv5->forward(conv4_output);

    auto pooled_output = adaptive_pool->forward(conv5_output);
    auto flattened_output = pooled_output.view({-1, 20}); // 20 是通道数

    auto fc1_output = torch::relu(fc1->forward(flattened_output));
    auto fc2_output = fc2->forward(fc1_output);
    return fc2_output;
}

// Model::Model() {
//     // 初始化卷积层，减少层数和每个层的通道数
//     conv1 = register_module("conv1", torch::nn::Sequential(
//         torch::nn::Conv1d(torch::nn::Conv1dOptions(1, 16, 3).stride(1).padding(1)),
//         torch::nn::BatchNorm1d(16),
//         torch::nn::ReLU(true),
//         torch::nn::MaxPool1d(2)
//     ));

//     conv2 = register_module("conv2", torch::nn::Sequential(
//         torch::nn::Conv1d(torch::nn::Conv1dOptions(16, 32, 3).stride(1).padding(1)),
//         torch::nn::BatchNorm1d(32),
//         torch::nn::ReLU(true),
//         torch::nn::MaxPool1d(2)
//     ));

//     conv3 = register_module("conv3", torch::nn::Sequential(
//         torch::nn::Conv1d(torch::nn::Conv1dOptions(32, 64, 3).stride(1).padding(1)),
//         torch::nn::BatchNorm1d(64),
//         torch::nn::ReLU(true),
//         torch::nn::MaxPool1d(2)
//     ));

//     // 初始化自适应池化层
//     adaptive_pool = register_module("adaptive_pool", torch::nn::AdaptiveAvgPool1d(1));

//     // 初始化全连接层，减少节点数
//     fc = register_module("fc", torch::nn::Sequential(
//         torch::nn::Dropout(0.5),
//         torch::nn::Linear(64, 32), // 减少到32个节点
//         torch::nn::BatchNorm1d(32),
//         torch::nn::ReLU(true),
//         torch::nn::Dropout(0.5),
//         torch::nn::Linear(32, 2) // 输出为2个类别
//     ));
// }

// torch::Tensor Model::forward(torch::Tensor input) {
//     // 依次通过卷积层
//     input = conv1->forward(input);
//     input = conv2->forward(input);
//     input = conv3->forward(input);

//     // 通过自适应平均池化层
//     input = adaptive_pool->forward(input);

//     // 展平输出
//     input = input.view({-1, 64}); // 64 是通道数

//     // 通过全连接层
//     input = fc->forward(input);

//     return input;
// }

// Model::Model() {
//     // 初始化卷积层，全部使用3x3卷积核并添加额外的卷积层以增加深度
//     conv1 = register_module("conv1", torch::nn::Sequential(
//         torch::nn::Conv1d(torch::nn::Conv1dOptions(1, 32, 3).stride(1).padding(1)),
//         torch::nn::BatchNorm1d(32),
//         torch::nn::ReLU(true),
//         torch::nn::Conv1d(torch::nn::Conv1dOptions(32, 32, 3).stride(1).padding(1)),
//         torch::nn::BatchNorm1d(32),
//         torch::nn::ReLU(true),
//         torch::nn::MaxPool1d(2)
//     ));

//     conv2 = register_module("conv2", torch::nn::Sequential(
//         torch::nn::Conv1d(torch::nn::Conv1dOptions(32, 64, 3).stride(1).padding(1)),
//         torch::nn::BatchNorm1d(64),
//         torch::nn::ReLU(true),
//         torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 64, 3).stride(1).padding(1)),
//         torch::nn::BatchNorm1d(64),
//         torch::nn::ReLU(true),
//         torch::nn::MaxPool1d(2)
//     ));

//     conv3 = register_module("conv3", torch::nn::Sequential(
//         torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 128, 3).stride(1).padding(1)),
//         torch::nn::BatchNorm1d(128),
//         torch::nn::ReLU(true),
//         torch::nn::Conv1d(torch::nn::Conv1dOptions(128, 128, 3).stride(1).padding(1)),
//         torch::nn::BatchNorm1d(128),
//         torch::nn::ReLU(true),
//         torch::nn::MaxPool1d(2)
//     ));

//     conv4 = register_module("conv4", torch::nn::Sequential(
//         torch::nn::Conv1d(torch::nn::Conv1dOptions(128, 256, 3).stride(1).padding(1)),
//         torch::nn::BatchNorm1d(256),
//         torch::nn::ReLU(true),
//         torch::nn::Conv1d(torch::nn::Conv1dOptions(256, 256, 3).stride(1).padding(1)),
//         torch::nn::BatchNorm1d(256),
//         torch::nn::ReLU(true),
//         torch::nn::MaxPool1d(2)
//     ));

//     conv5 = register_module("conv5", torch::nn::Sequential(
//         torch::nn::Conv1d(torch::nn::Conv1dOptions(256, 256, 3).stride(1).padding(1)),
//         torch::nn::BatchNorm1d(256),
//         torch::nn::ReLU(true),
//         torch::nn::Conv1d(torch::nn::Conv1dOptions(256, 256, 3).stride(1).padding(1)),
//         torch::nn::BatchNorm1d(256),
//         torch::nn::ReLU(true),
//         torch::nn::MaxPool1d(2)
//     ));

//     // 初始化自适应池化层
//     adaptive_pool = register_module("adaptive_pool", torch::nn::AdaptiveAvgPool1d(1));

//     // 初始化全连接层
//     fc = register_module("fc", torch::nn::Sequential(
//         torch::nn::Dropout(0.5),
//         torch::nn::Linear(256, 128), // 256 是通道数
//         torch::nn::BatchNorm1d(128),
//         torch::nn::ReLU(true),
//         torch::nn::Dropout(0.5),
//         torch::nn::Linear(128, 2) // 输出为2个类别
//     ));
// }

// torch::Tensor Model::forward(torch::Tensor input) {
//     // 依次通过卷积层
//     input = conv1->forward(input);
//     input = conv2->forward(input);
//     input = conv3->forward(input);
//     input = conv4->forward(input);
//     input = conv5->forward(input);

//     // 通过自适应平均池化层
//     input = adaptive_pool->forward(input);

//     // 展平输出
//     input = input.view({-1, 256}); // 256 是通道数

//     // 通过全连接层
//     input = fc->forward(input);

//     return input;
// }

// Model::Model() {
//     // 初始化卷积层，使用更深更宽的网络并加入残差连接
//     conv1 = register_module("conv1", make_residual_block(1, 32, 7, 2));
//     downsample1 = register_module("downsample1", make_downsample(1, 32, 2));

//     conv2 = register_module("conv2", make_residual_block(32, 64, 5, 2));
//     downsample2 = register_module("downsample2", make_downsample(32, 64, 2));

//     conv3 = register_module("conv3", make_residual_block(64, 128, 3, 2));
//     downsample3 = register_module("downsample3", make_downsample(64, 128, 2));

//     conv4 = register_module("conv4", make_residual_block(128, 256, 3, 2));
//     downsample4 = register_module("downsample4", make_downsample(128, 256, 2));

//     conv5 = register_module("conv5", make_residual_block(256, 256, 3, 2));
//     downsample5 = register_module("downsample5", make_downsample(256, 256, 2));

//     // 初始化自适应池化层
//     adaptive_pool = register_module("adaptive_pool", torch::nn::AdaptiveAvgPool1d(1));

//     // 初始化全连接层
//     fc1 = register_module("fc1", torch::nn::Sequential(
//         torch::nn::Dropout(0.5),
//         torch::nn::Linear(256, 128), // 256 是通道数
//         torch::nn::BatchNorm1d(128),
//         torch::nn::ReLU(true),
//         torch::nn::Dropout(0.5),
//         torch::nn::Linear(128, 64)
//     ));

//     fc2 = register_module("fc2", torch::nn::Linear(64, 2)); // 64 是 fc1 的输出
// }

// torch::Tensor Model::forward(torch::Tensor input) {
//     // 添加残差连接，并确保尺寸匹配
//     auto conv1_output = conv1->forward(input);
//     auto residual1 = downsample1->forward(input);
//     if (conv1_output.size() != residual1.size()) {
//         // 如果尺寸不匹配，尝试通过填充或其他方式调整尺寸
//         // 这里假设是最后一个维度不匹配，可以根据实际情况调整
//         residual1 = residual1.view({residual1.size(0), residual1.size(1), conv1_output.size(2)});
//     }
//     input = conv1_output + residual1;
//     input = torch::relu_(input);

//     auto conv2_output = conv2->forward(input);
//     auto residual2 = downsample2->forward(input);
//     if (conv2_output.size() != residual2.size()) {
//         residual2 = residual2.view({residual2.size(0), residual2.size(1), conv2_output.size(2)});
//     }
//     input = conv2_output + residual2;
//     input = torch::relu_(input);

//     auto conv3_output = conv3->forward(input);
//     auto residual3 = downsample3->forward(input);
//     if (conv3_output.size() != residual3.size()) {
//         residual3 = residual3.view({residual3.size(0), residual3.size(1), conv3_output.size(2)});
//     }
//     input = conv3_output + residual3;
//     input = torch::relu_(input);

//     auto conv4_output = conv4->forward(input);
//     auto residual4 = downsample4->forward(input);
//     if (conv4_output.size() != residual4.size()) {
//         residual4 = residual4.view({residual4.size(0), residual4.size(1), conv4_output.size(2)});
//     }
//     input = conv4_output + residual4;
//     input = torch::relu_(input);

//     auto conv5_output = conv5->forward(input);
//     auto residual5 = downsample5->forward(input);
//     if (conv5_output.size() != residual5.size()) {
//         residual5 = residual5.view({residual5.size(0), residual5.size(1), conv5_output.size(2)});
//     }
//     input = conv5_output + residual5;
//     input = torch::relu_(input);

//     auto pooled_output = adaptive_pool->forward(input);
//     auto flattened_output = pooled_output.view({-1, 256}); // 256 是通道数

//     auto fc1_output = fc1->forward(flattened_output);
//     auto fc2_output = fc2->forward(fc1_output);
//     return fc2_output;
// }

// Model::Model()
// {
//     // 初始化卷积层
//     conv1 = register_module("conv1", torch::nn::Sequential(
//                                          torch::nn::Conv1d(torch::nn::Conv1dOptions(1, 32, 3).stride(2).padding(1)),
//                                          torch::nn::ReLU(true),
//                                          torch::nn::BatchNorm1d(32)));

//     conv2 = register_module("conv2", torch::nn::Sequential(
//                                          torch::nn::Conv1d(torch::nn::Conv1dOptions(32, 64, 3).stride(2).padding(1)),
//                                          torch::nn::ReLU(true),
//                                          torch::nn::BatchNorm1d(64)));

//     // 初始化自适应池化层
//     adaptive_pool = register_module("adaptive_pool", torch::nn::AdaptiveAvgPool1d(1));

//     // 初始化全连接层
//     fc1 = register_module("fc1", torch::nn::Sequential(
//                                      torch::nn::Dropout(0.5),
//                                      torch::nn::Linear(64, 32) // 64 是通道数
//                                      ));

//     fc2 = register_module("fc2", torch::nn::Linear(32, 2)); // 32 是 fc1 的输出
// }

// torch::Tensor Model::forward(torch::Tensor input)
// {
//     auto conv1_output = conv1->forward(input);
//     auto conv2_output = conv2->forward(conv1_output);

//     auto pooled_output = adaptive_pool->forward(conv2_output);
//     auto flattened_output = pooled_output.view({-1, 64}); // 64 是通道数

//     auto fc1_output = torch::relu(fc1->forward(flattened_output));
//     auto fc2_output = fc2->forward(fc1_output);
//     return fc2_output;
// }

/*
torch::Tensor Model::forward(torch::Tensor input)
{
  auto conv1_output = conv1->forward(input);
  std::cout << "conv1 output shape: " << conv1_output.sizes() << std::endl;

  auto conv2_output = conv2->forward(conv1_output);
  std::cout << "conv2 output shape: " << conv2_output.sizes() << std::endl;

  auto conv3_output = conv3->forward(conv2_output);
  std::cout << "conv3 output shape: " << conv3_output.sizes() << std::endl;

  auto conv4_output = conv4->forward(conv3_output);
  std::cout << "conv4 output shape: " << conv4_output.sizes() << std::endl;

  auto conv5_output = conv5->forward(conv4_output);
  std::cout << "conv5 output shape: " << conv5_output.sizes() << std::endl;

  // 使用自适应池化层
  auto pooled_output = adaptive_pool->forward(conv5_output);

  // 自适应池化后的输出形状应为 [batch_size, channels, 1]
  // 因此展平后的形状为 [batch_size, channels]
  auto flattened_output = pooled_output.view({-1, 20}); // 注意这里的20是通道数

  auto fc1_output = torch::relu(fc1->forward(flattened_output));
  auto fc2_output = fc2->forward(fc1_output);
  std::cout << "fc2 output shape: " << fc2_output.sizes() << std::endl;
  return fc2_output;
}*/