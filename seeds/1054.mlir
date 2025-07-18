module {
  func.func @main(%arg0: tensor<i16>, %arg1: tensor<i16>, %arg2: tensor<2x78x56x89xf32>, %arg3: tensor<77x7x4x8xf32>, %arg4: tensor<77xf32>, %arg5: tensor<13x42xi1>) -> (tensor<i16>, tensor<13x42xi1>, tensor<13x42xi1>, tensor<2x163x61x77xi1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 1>, out_shape = array<i64: 2, 163, 61, 77>} : (tensor<2x78x56x89xf32>, tensor<77x7x4x8xf32>, tensor<77xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x163x61x77xf32>
    %2 = tosa.logical_not %arg5 : (tensor<13x42xi1>) -> tensor<13x42xi1>
    %3 = tosa.greater_equal %1, %1 : (tensor<2x163x61x77xf32>, tensor<2x163x61x77xf32>) -> tensor<2x163x61x77xi1>
    %4 = tosa.clz %2 : (tensor<13x42xi1>) -> tensor<13x42xi1>
    %5 = tosa.bitwise_and %2, %2 : (tensor<13x42xi1>, tensor<13x42xi1>) -> tensor<13x42xi1>
    %6 = tosa.logical_right_shift %3, %3 : (tensor<2x163x61x77xi1>, tensor<2x163x61x77xi1>) -> tensor<2x163x61x77xi1>
    return %0, %4, %5, %6 : tensor<i16>, tensor<13x42xi1>, tensor<13x42xi1>, tensor<2x163x61x77xi1>
  }
}
