module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<3x3x67x54xf32>, %arg2: tensor<28x34x87x6xf32>, %arg3: tensor<28xf32>, %arg4: tensor<46x79xi1>, %arg5: tensor<1x79xi1>) -> (tensor<f32>, tensor<1x11x1x10xf32>, tensor<6x42x221x28xf32>, tensor<46x1xi1>) {
    %0 = tosa.abs %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.floor %0 : (tensor<f32>) -> tensor<f32>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 1>, stride = array<i64: 2, 2>, out_shape = array<i64: 3, 42, 221, 28>} : (tensor<3x3x67x54xf32>, tensor<28x34x87x6xf32>, tensor<28xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3x42x221x28xf32>
    %s_3_start = tosa.const_shape {values = dense<[ 0, 2, 2, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_3_size = tosa.const_shape {values = dense<[ 12, 12, 1, 4 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.slice %2, %s_3_start, %s_3_size : (tensor<3x42x221x28xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<12x12x1x4xf32>
    %4 = tosa.logical_xor %arg4, %arg5 : (tensor<46x79xi1>, tensor<1x79xi1>) -> tensor<46x79xi1>
    %s_5_start = tosa.const_shape {values = dense<[ 1, 1, 0, 0 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_5_size = tosa.const_shape {values = dense<[ 1, 11, 1, 10 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %5 = tosa.slice %3, %s_5_start, %s_5_size : (tensor<12x12x1x4xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x11x1x10xf32>
    %6 = tosa.floor %5 : (tensor<1x11x1x10xf32>) -> tensor<1x11x1x10xf32>
    %7 = tosa.abs %2 : (tensor<3x42x221x28xf32>) -> tensor<3x42x221x28xf32>
    %8 = tosa.sub %7, %7 : (tensor<3x42x221x28xf32>, tensor<3x42x221x28xf32>) -> tensor<3x42x221x28xf32>
    %9 = tosa.concat %8, %7 {axis = 0 : i32} : (tensor<3x42x221x28xf32>, tensor<3x42x221x28xf32>) -> tensor<6x42x221x28xf32>
    %10 = tosa.reduce_any %4 {axis = 1 : i32} : (tensor<46x79xi1>) -> tensor<46x1xi1>
    return %1, %6, %9, %10 : tensor<f32>, tensor<1x11x1x10xf32>, tensor<6x42x221x28xf32>, tensor<46x1xi1>
  }
}
