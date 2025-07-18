module {
  func.func @main(%arg0: tensor<97xi1>, %arg1: tensor<1xi1>, %arg2: tensor<50x48x1x16xf32>, %arg3: tensor<2x62x44x62xf32>, %arg4: tensor<2xf32>) -> (tensor<97xi1>, tensor<50x113x46x2xf32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<97xi1>, tensor<1xi1>) -> tensor<97xi1>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 1>, stride = array<i64: 1, 2>, out_shape = array<i64: 50, 113, 46, 2>} : (tensor<50x48x1x16xf32>, tensor<2x62x44x62xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<50x113x46x2xf32>
    %2 = tosa.sigmoid %1 : (tensor<50x113x46x2xf32>) -> tensor<50x113x46x2xf32>
    %3 = tosa.reciprocal %2 : (tensor<50x113x46x2xf32>) -> tensor<50x113x46x2xf32>
    %4 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<97xi1>, tensor<97xi1>) -> tensor<97xi1>
    %5 = tosa.log %3 : (tensor<50x113x46x2xf32>) -> tensor<50x113x46x2xf32>
    return %4, %5 : tensor<97xi1>, tensor<50x113x46x2xf32>
  }
}
