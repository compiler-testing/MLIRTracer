module {
  func.func @main(%arg0: tensor<42x84x85xi32>, %arg1: tensor<75x48x73x62xf32>, %arg2: tensor<64x45x67x68xf32>, %arg3: tensor<64xf32>, %arg4: tensor<31x86x95x10x80x72xi1>) -> (tensor<42x84x1xi32>, tensor<31x86x95x10x80x72xi1>, tensor<1x142x1x64xf32>) {
    %0 = tosa.reduce_sum %arg0 {axis = 2 : i32} : (tensor<42x84x85xi32>) -> tensor<42x84x1xi32>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 2, 1, 2, 1>, stride = array<i64: 2, 2>, out_shape = array<i64: 75, 142, 214, 64>} : (tensor<75x48x73x62xf32>, tensor<64x45x67x68xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<75x142x214x64xf32>
    %2 = tosa.bitwise_or %0, %0 : (tensor<42x84x1xi32>, tensor<42x84x1xi32>) -> tensor<42x84x1xi32>
    %3 = tosa.identity %1 : (tensor<75x142x214x64xf32>) -> tensor<75x142x214x64xf32>
    %4 = tosa.reverse %2 {axis = 2 : i32} : (tensor<42x84x1xi32>) -> tensor<42x84x1xi32>
    %5 = tosa.identity %4 : (tensor<42x84x1xi32>) -> tensor<42x84x1xi32>
    %6 = tosa.add %5, %2 : (tensor<42x84x1xi32>, tensor<42x84x1xi32>) -> tensor<42x84x1xi32>
    %7 = tosa.reduce_min %3 {axis = 0 : i32} : (tensor<75x142x214x64xf32>) -> tensor<1x142x214x64xf32>
    %8 = tosa.bitwise_not %6 : (tensor<42x84x1xi32>) -> tensor<42x84x1xi32>
    %9 = tosa.logical_right_shift %8, %8 : (tensor<42x84x1xi32>, tensor<42x84x1xi32>) -> tensor<42x84x1xi32>
    %10 = tosa.logical_not %arg4 : (tensor<31x86x95x10x80x72xi1>) -> tensor<31x86x95x10x80x72xi1>
    %11 = tosa.tanh %7 : (tensor<1x142x214x64xf32>) -> tensor<1x142x214x64xf32>
    %12 = tosa.reduce_sum %11 {axis = 2 : i32} : (tensor<1x142x214x64xf32>) -> tensor<1x142x1x64xf32>
    return %9, %10, %12 : tensor<42x84x1xi32>, tensor<31x86x95x10x80x72xi1>, tensor<1x142x1x64xf32>
  }
}
