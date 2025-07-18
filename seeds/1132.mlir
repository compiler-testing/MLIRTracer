module {
  func.func @main(%arg0: tensor<7x95x99xi8>, %arg1: tensor<7x1x1xi8>, %arg2: tensor<43xi1>, %arg3: tensor<67x75x55x77xf32>, %arg4: tensor<37x43x52x11xf32>, %arg5: tensor<37xf32>) -> (tensor<67x121x163x37xf32>, tensor<1xi1>, tensor<7x95x99xi8>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<7x95x99xi8>, tensor<7x1x1xi8>) -> tensor<7x95x99xi8>
    %1 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<43xi1>) -> tensor<1xi1>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg3, %arg4, %arg5, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 1>, stride = array<i64: 1, 2>, out_shape = array<i64: 67, 121, 163, 37>} : (tensor<67x75x55x77xf32>, tensor<37x43x52x11xf32>, tensor<37xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<67x121x163x37xf32>
    %3 = tosa.add %2, %2 : (tensor<67x121x163x37xf32>, tensor<67x121x163x37xf32>) -> tensor<67x121x163x37xf32>
    %4 = tosa.logical_xor %1, %1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.maximum %0, %0 : (tensor<7x95x99xi8>, tensor<7x95x99xi8>) -> tensor<7x95x99xi8>
    return %3, %4, %5 : tensor<67x121x163x37xf32>, tensor<1xi1>, tensor<7x95x99xi8>
  }
}
