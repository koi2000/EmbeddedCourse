#ifndef MODEL_H  // 检查宏 MODEL_H 是否未定义
#define MODEL_H  // 定义宏 MODEL_H
#include <torch/torch.h>
#include <torch/serialize/archive.h>
#include <iostream>

class Model : public torch::nn::Module {
public:
    // 构造函数
    Model();

    // 前向传播
    torch::Tensor forward(torch::Tensor input);

private:
    // 卷积层
    torch::nn::Sequential conv1, conv2, conv3, conv4, conv5;
    // 全连接层
    torch::nn::Sequential fc1;
    torch::nn::Linear fc2 = nullptr; 
    torch::nn::AdaptiveAvgPool1d adaptive_pool = nullptr; // 自适应池化层
};

// class Model : public torch::nn::Module {
// public:
//     // 构造函数
//     Model();

//     // 前向传播
//     torch::Tensor forward(torch::Tensor input);

// private:
//     // 卷积层
//     torch::nn::Sequential conv1, conv2, conv3;
//     // 自适应池化层
//     torch::nn::AdaptiveAvgPool1d adaptive_pool = nullptr;
//     // 全连接层
//     torch::nn::Sequential fc;
// };

// class Model : public torch::nn::Module {
// public:
//     // 构造函数
//     Model();

//     // 前向传播
//     torch::Tensor forward(torch::Tensor input);

// private:
//     // 卷积层
//     torch::nn::Sequential conv1, conv2, conv3, conv4, conv5;
//     // 自适应池化层
//     torch::nn::AdaptiveAvgPool1d adaptive_pool = nullptr;
//     // 全连接层
//     torch::nn::Sequential fc;
// };

// class Model : public torch::nn::Module {
// public:
//     // 构造函数
//     Model();

//     // 前向传播
//     torch::Tensor forward(torch::Tensor input);

// private:
//     // 卷积层
//     torch::nn::Sequential conv1, conv2, conv3, conv4, conv5;
//     // 下采样层
//     torch::nn::Sequential downsample1, downsample2, downsample3, downsample4, downsample5;
//     // 全连接层
//     torch::nn::Sequential fc1;
//     torch::nn::Linear fc2 = nullptr;
//     // 自适应池化层
//     torch::nn::AdaptiveAvgPool1d adaptive_pool = nullptr;

//     // 残差块定义
//     torch::nn::Sequential make_residual_block(int in_channels, int out_channels, int kernel_size, int stride) {
//         return torch::nn::Sequential(
//             torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size).stride(stride).padding(1)),
//             torch::nn::BatchNorm1d(out_channels),
//             torch::nn::ReLU(true),
//             torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels, out_channels, 3).stride(1).padding(1)),
//             torch::nn::BatchNorm1d(out_channels)
//         );
//     }

//     // 下采样定义，确保输出尺寸匹配
//     torch::nn::Sequential make_downsample(int in_channels, int out_channels, int stride, int padding = 0) {
//         return torch::nn::Sequential(
//             torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, 1).stride(stride).padding(padding))
//         );
//     }
// };

// class Model : public torch::nn::Module {
// public:
//     // 构造函数
//     Model();

//     // 前向传播
//     torch::Tensor forward(torch::Tensor input);

// private:
//     torch::nn::Sequential conv1, conv2;
//     torch::nn::AdaptiveAvgPool1d adaptive_pool = nullptr;
//     torch::nn::Sequential fc1;
//     torch::nn::Linear fc2 = nullptr;
// };
#endif