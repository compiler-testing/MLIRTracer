module {
  func.func @main(%arg0: tensor<96x51x93xi8>, %arg1: tensor<36x89x94x65xf32>, %arg2: tensor<46x27x11x82xf32>, %arg3: tensor<46xf32>, %arg4: tensor<31x19x59x97xi1>, %arg5: tensor<31x1x1x97xi1>) -> (tensor<36x205x106x46xf32>, tensor<96x1x93xi8>, tensor<31x19x59x97xi1>) {
    %0 = tosa.reduce_max %arg0 {axis = 1 : i32} : (tensor<96x51x93xi8>) -> tensor<96x1x93xi8>
    %1 = tosa.sigmoid %arg1 : (tensor<36x89x94x65xf32>) -> tensor<36x89x94x65xf32>
    %2 = tosa.exp %1 : (tensor<36x89x94x65xf32>) -> tensor<36x89x94x65xf32>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %2, %arg2, %arg3, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 1>, out_shape = array<i64: 36, 205, 106, 46>} : (tensor<36x89x94x65xf32>, tensor<46x27x11x82xf32>, tensor<46xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<36x205x106x46xf32>
    %4 = tosa.bitwise_and %0, %0 : (tensor<96x1x93xi8>, tensor<96x1x93xi8>) -> tensor<96x1x93xi8>
    %5 = tosa.bitwise_or %0, %4 : (tensor<96x1x93xi8>, tensor<96x1x93xi8>) -> tensor<96x1x93xi8>
    %6 = tosa.logical_and %arg4, %arg5 : (tensor<31x19x59x97xi1>, tensor<31x1x1x97xi1>) -> tensor<31x19x59x97xi1>
    return %3, %5, %6 : tensor<36x205x106x46xf32>, tensor<96x1x93xi8>, tensor<31x19x59x97xi1>
  }
}
