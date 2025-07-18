module {
  func.func @main(%arg0: tensor<41x70x34x40x27x5xf32>, %arg1: tensor<9x41x68x1xf32>, %arg2: tensor<74x10x58x42xi1>) -> (tensor<1x3x4x11x12x5xf32>, tensor<74x1x58x42xi1>, tensor<1x82x204x1xf32>, tensor<74x1x58x42xi1>, tensor<1x41x68x1xf32>) {
    %0 = tosa.abs %arg0 : (tensor<41x70x34x40x27x5xf32>) -> tensor<41x70x34x40x27x5xf32>
    %1 = tosa.reduce_product %arg1 {axis = 0 : i32} : (tensor<9x41x68x1xf32>) -> tensor<1x41x68x1xf32>
    %2 = tosa.reduce_all %arg2 {axis = 1 : i32} : (tensor<74x10x58x42xi1>) -> tensor<74x1x58x42xi1>
    %s_3_start = tosa.const_shape {values = dense<[ 13, 14, 14, 28, 6, 0 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_3_size = tosa.const_shape {values = dense<[ 1, 3, 4, 11, 12, 5 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %3 = tosa.slice %0, %s_3_start, %s_3_size : (tensor<41x70x34x40x27x5xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<1x3x4x11x12x5xf32>
    %4 = tosa.clamp %1 {min_val = 3.800000e+01 : f32, max_val = 4.300000e+01 : f32} : (tensor<1x41x68x1xf32>) -> tensor<1x41x68x1xf32>
    %5 = tosa.logical_right_shift %2, %2 : (tensor<74x1x58x42xi1>, tensor<74x1x58x42xi1>) -> tensor<74x1x58x42xi1>
    %t_6 = tosa.const_shape {values = dense<[ 1, 2, 3, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %6 = tosa.tile %4, %t_6 : (tensor<1x41x68x1xf32>, !tosa.shape<4>) -> tensor<1x82x204x1xf32>
    %7 = tosa.logical_or %2, %2 : (tensor<74x1x58x42xi1>, tensor<74x1x58x42xi1>) -> tensor<74x1x58x42xi1>
    %8 = tosa.log %1 : (tensor<1x41x68x1xf32>) -> tensor<1x41x68x1xf32>
    %9 = tosa.maximum %8, %8 : (tensor<1x41x68x1xf32>, tensor<1x41x68x1xf32>) -> tensor<1x41x68x1xf32>
    return %3, %5, %6, %7, %9 : tensor<1x3x4x11x12x5xf32>, tensor<74x1x58x42xi1>, tensor<1x82x204x1xf32>, tensor<74x1x58x42xi1>, tensor<1x41x68x1xf32>
  }
}
