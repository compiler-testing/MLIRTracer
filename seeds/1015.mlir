module {
  func.func @main(%arg0: tensor<3x77x31x61xf32>, %arg1: tensor<3x94x10x67xf32>, %arg2: tensor<3xf32>, %arg3: tensor<31xi1>) -> (tensor<3x496x72x3xf32>, tensor<3x248x72x3xf32>, tensor<1xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>, out_shape = array<i64: 3, 248, 72, 3>} : (tensor<3x77x31x61xf32>, tensor<3x94x10x67xf32>, tensor<3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3x248x72x3xf32>
    %1 = tosa.clz %arg3 : (tensor<31xi1>) -> tensor<31xi1>
    %2 = tosa.concat %0, %0 {axis = 1 : i32} : (tensor<3x248x72x3xf32>, tensor<3x248x72x3xf32>) -> tensor<3x496x72x3xf32>
    %3 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<31xi1>) -> tensor<1xi1>
    %4 = tosa.bitwise_not %3 : (tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.arithmetic_right_shift %3, %3 {round = false} : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.pow %0, %0 : (tensor<3x248x72x3xf32>, tensor<3x248x72x3xf32>) -> tensor<3x248x72x3xf32>
    %7 = tosa.bitwise_xor %4, %5 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %2, %6, %7 : tensor<3x496x72x3xf32>, tensor<3x248x72x3xf32>, tensor<1xi1>
  }
}
