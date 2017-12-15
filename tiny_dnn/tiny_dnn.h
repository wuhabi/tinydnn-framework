/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/config.h"
#include "tiny_dnn/network.h"
#include "tiny_dnn/nodes.h"

#include "tiny_dnn/core/framework/tensor.h"

#include "tiny_dnn/core/framework/device.h"
#include "tiny_dnn/core/framework/program_manager.h"

#include "tiny_dnn/layers/arithmetic_layer.h"
#include "tiny_dnn/layers/average_pooling_layer.h"
#include "tiny_dnn/layers/average_unpooling_layer.h"
#include "tiny_dnn/layers/batch_normalization_layer.h"
#include "tiny_dnn/layers/concat_layer.h"
#include "tiny_dnn/layers/convolutional_layer.h"
#include "tiny_dnn/layers/deconvolutional_layer.h"
#include "tiny_dnn/layers/dropout_layer.h"
#include "tiny_dnn/layers/feedforward_layer.h"
#include "tiny_dnn/layers/fully_connected_layer.h"
#include "tiny_dnn/layers/input_layer.h"
#include "tiny_dnn/layers/linear_layer.h"
#include "tiny_dnn/layers/lrn_layer.h"
#include "tiny_dnn/layers/max_pooling_layer.h"
#include "tiny_dnn/layers/max_unpooling_layer.h"
#include "tiny_dnn/layers/power_layer.h"
#include "tiny_dnn/layers/quantized_convolutional_layer.h"
#include "tiny_dnn/layers/quantized_deconvolutional_layer.h"
#include "tiny_dnn/layers/slice_layer.h"

#ifdef CNN_USE_GEMMLOWP
#include "tiny_dnn/layers/quantized_fully_connected_layer.h"
#endif  // CNN_USE_GEMMLOWP

#include "tiny_dnn/activations/activation_function.h"
#include "tiny_dnn/lossfunctions/loss_function.h"
#include "tiny_dnn/optimizers/optimizer.h"

#include "tiny_dnn/util/deform.h"
#include "tiny_dnn/util/graph_visualizer.h"
#include "tiny_dnn/util/product.h"
#include "tiny_dnn/util/weight_init.h"

#include "tiny_dnn/io/cifar10_parser.h"
#include "tiny_dnn/io/display.h"
#include "tiny_dnn/io/layer_factory.h"
#include "tiny_dnn/io/mnist_parser.h"

#ifdef DNN_USE_IMAGE_API
#include "tiny_dnn/util/image.h"
#endif  // DNN_USE_IMAGE_API

#ifndef CNN_NO_SERIALIZATION
#include "tiny_dnn/util/deserialization_helper.h"
#include "tiny_dnn/util/serialization_helper.h"
#endif  // CNN_NO_SERIALIZATION

#ifdef CNN_USE_CAFFE_CONVERTER
// experimental / require google protobuf
#include "tiny_dnn/io/caffe/layer_factory.h"
#endif

// shortcut version of layer names
namespace tiny_dnn {
namespace layers {

template <class T>
using conv = tiny_dnn::convolutional_layer<T>;

template <class T>
using q_conv = tiny_dnn::quantized_convolutional_layer<T>;

template <class T>
using max_pool = tiny_dnn::max_pooling_layer<T>;

template <class T>
using ave_pool = tiny_dnn::average_pooling_layer<T>;

template <class T>
using fc = tiny_dnn::fully_connected_layer<T>;

template <class T>
using dense = tiny_dnn::fully_connected_layer<T>;

using add = tiny_dnn::elementwise_add_layer;

using dropout = tiny_dnn::dropout_layer;

using input = tiny_dnn::input_layer;

template <class T>
using lrn = tiny_dnn::lrn_layer<T>;

using input = tiny_dnn::input_layer;

using concat = tiny_dnn::concat_layer;

template <class T>
using deconv = tiny_dnn::deconvolutional_layer<T>;

template <class T>
using max_unpool = tiny_dnn::max_unpooling_layer<T>;

template <class T>
using ave_unpool = tiny_dnn::average_unpooling_layer<T>;

}  // namespace layers

#include "tiny_dnn/models/alexnet.h"

using batch_norm = tiny_dnn::batch_normalization_layer;

using slice = tiny_dnn::slice_layer;

using power = tiny_dnn::power_layer;

}  // namespace tiny_dnn