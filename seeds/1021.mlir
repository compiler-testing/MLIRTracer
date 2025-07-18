module {
  func.func @main(%arg0: tensor<5x17x61xi32>, %arg1: tensor<1x1x61xi32>, %arg2: tensor<59x83x98x90xf32>, %arg3: tensor<4x8x25x31xf32>, %arg4: tensor<4xf32>, %arg5: tensor<97xi1>) -> (tensor<5x1x61xi32>, tensor<1xi1>, tensor<59x92x124x4xf32>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<5x17x61xi32>, tensor<1x1x61xi32>) -> tensor<5x17x61xi32>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 59, 92, 124, 4>} : (tensor<59x83x98x90xf32>, tensor<4x8x25x31xf32>, tensor<4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<59x92x124x4xf32>
    %2 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<5x17x61xi32>) -> tensor<5x1x61xi32>
    %3 = tosa.reduce_all %arg5 {axis = 0 : i32} : (tensor<97xi1>) -> tensor<1xi1>
    %4 = tosa.logical_xor %3, %3 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.logical_or %4, %4 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.ceil %1 : (tensor<59x92x124x4xf32>) -> tensor<59x92x124x4xf32>
    return %2, %5, %6 : tensor<5x1x61xi32>, tensor<1xi1>, tensor<59x92x124x4xf32>
  }
}
