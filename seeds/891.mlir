module {
  func.func @main(%arg0: tensor<54x74x12xi1>, %arg1: tensor<35x16x11x15xf32>, %arg2: tensor<97x7x51x56xf32>, %arg3: tensor<97xf32>) -> (tensor<54x74x12xi1>, tensor<35x74x97xi32>) {
    %0 = tosa.logical_not %arg0 : (tensor<54x74x12xi1>) -> tensor<54x74x12xi1>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<54x74x12xi1>, tensor<54x74x12xi1>) -> tensor<54x74x12xi1>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 1, 1, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 35, 40, 74, 97>} : (tensor<35x16x11x15xf32>, tensor<97x7x51x56xf32>, tensor<97xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<35x40x74x97xf32>
    %3 = tosa.sub %2, %2 : (tensor<35x40x74x97xf32>, tensor<35x40x74x97xf32>) -> tensor<35x40x74x97xf32>
    %4 = tosa.argmax %3 {axis = 1 : i32} : (tensor<35x40x74x97xf32>) -> tensor<35x74x97xi32>
    return %1, %4 : tensor<54x74x12xi1>, tensor<35x74x97xi32>
  }
}
