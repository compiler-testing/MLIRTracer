module {
  func.func @main(%arg0: tensor<75xf32>, %arg1: tensor<75xf32>, %arg2: tensor<31x39x31x40xf32>, %arg3: tensor<77x74x45x27xf32>, %arg4: tensor<77xf32>) -> (tensor<75xi1>, tensor<31x153x79x77xf32>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<75xf32>, tensor<75xf32>) -> tensor<75xi1>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 2, 2, 2>, stride = array<i64: 2, 1>, out_shape = array<i64: 31, 153, 79, 77>} : (tensor<31x39x31x40xf32>, tensor<77x74x45x27xf32>, tensor<77xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<31x153x79x77xf32>
    return %0, %1 : tensor<75xi1>, tensor<31x153x79x77xf32>
  }
}
