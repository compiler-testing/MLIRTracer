module {
  func.func @main(%arg0: tensor<85x5x47x42xf32>, %arg1: tensor<65x69x63x5xf32>, %arg2: tensor<65xf32>, %arg3: tensor<77xi1>) -> (tensor<77xi1>, tensor<77xi1>, tensor<85x80x113x65xf32>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 1, 2, 2>, stride = array<i64: 2, 1>, out_shape = array<i64: 85, 80, 113, 65>} : (tensor<85x5x47x42xf32>, tensor<65x69x63x5xf32>, tensor<65xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<85x80x113x65xf32>
    %1 = tosa.logical_not %arg3 : (tensor<77xi1>) -> tensor<77xi1>
    %2 = tosa.logical_right_shift %1, %1 : (tensor<77xi1>, tensor<77xi1>) -> tensor<77xi1>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %3 = tosa.negate %1, %in_zp_3, %out_zp_3 : (tensor<77xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<77xi1>
    %4 = tosa.identity %3 : (tensor<77xi1>) -> tensor<77xi1>
    %5 = tosa.sigmoid %0 : (tensor<85x80x113x65xf32>) -> tensor<85x80x113x65xf32>
    return %2, %4, %5 : tensor<77xi1>, tensor<77xi1>, tensor<85x80x113x65xf32>
  }
}
