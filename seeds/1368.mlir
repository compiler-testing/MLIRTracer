module {
  func.func @main(%arg0: tensor<16x80x2x92xf32>, %arg1: tensor<78x20x29x21xf32>, %arg2: tensor<78xf32>, %arg3: tensor<98x74x96x95x34xi64>, %arg4: tensor<8x85x30x93xi1>, %arg5: tensor<8x85x1x93xi1>) -> (tensor<8x85x30x93xi1>, tensor<16x102x34x78xf32>, tensor<1x78xi32>, tensor<98x148x96x95x34xi64>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 1, 2, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 16, 102, 34, 78>} : (tensor<16x80x2x92xf32>, tensor<78x20x29x21xf32>, tensor<78xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<16x102x34x78xf32>
    %1 = tosa.bitwise_not %arg3 : (tensor<98x74x96x95x34xi64>) -> tensor<98x74x96x95x34xi64>
    %2 = tosa.argmax %0 {axis = 0 : i32} : (tensor<16x102x34x78xf32>) -> tensor<102x34x78xi32>
    %3 = tosa.argmax %2 {axis = 0 : i32} : (tensor<102x34x78xi32>) -> tensor<34x78xi32>
    %4 = tosa.reduce_min %3 {axis = 0 : i32} : (tensor<34x78xi32>) -> tensor<1x78xi32>
    %5 = tosa.logical_xor %arg4, %arg5 : (tensor<8x85x30x93xi1>, tensor<8x85x1x93xi1>) -> tensor<8x85x30x93xi1>
    %in_zp_6 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_6 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = tosa.negate %4, %in_zp_6, %out_zp_6 : (tensor<1x78xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x78xi32>
    %7 = tosa.rsqrt %0 : (tensor<16x102x34x78xf32>) -> tensor<16x102x34x78xf32>
    %8 = tosa.intdiv %6, %4 : (tensor<1x78xi32>, tensor<1x78xi32>) -> tensor<1x78xi32>
    %9 = tosa.abs %8 : (tensor<1x78xi32>) -> tensor<1x78xi32>
    %10 = tosa.logical_right_shift %1, %1 : (tensor<98x74x96x95x34xi64>, tensor<98x74x96x95x34xi64>) -> tensor<98x74x96x95x34xi64>
    %11 = tosa.sub %10, %1 : (tensor<98x74x96x95x34xi64>, tensor<98x74x96x95x34xi64>) -> tensor<98x74x96x95x34xi64>
    %12 = tosa.concat %11, %11 {axis = 1 : i32} : (tensor<98x74x96x95x34xi64>, tensor<98x74x96x95x34xi64>) -> tensor<98x148x96x95x34xi64>
    return %5, %7, %9, %12 : tensor<8x85x30x93xi1>, tensor<16x102x34x78xf32>, tensor<1x78xi32>, tensor<98x148x96x95x34xi64>
  }
}
