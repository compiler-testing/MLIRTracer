module {
  func.func @main(%arg0: tensor<100x84x57x4xf32>, %arg1: tensor<87x54x57x32xf32>, %arg2: tensor<87xf32>, %arg3: tensor<28x96x45xi1>) -> (tensor<100x224x116x87xf32>, tensor<28x45xi32>, tensor<1x96x1xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 1>, stride = array<i64: 2, 1>, out_shape = array<i64: 100, 224, 116, 87>} : (tensor<100x84x57x4xf32>, tensor<87x54x57x32xf32>, tensor<87xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<100x224x116x87xf32>
    %1 = tosa.ceil %0 : (tensor<100x224x116x87xf32>) -> tensor<100x224x116x87xf32>
    %2 = tosa.logical_not %arg3 : (tensor<28x96x45xi1>) -> tensor<28x96x45xi1>
    %3 = tosa.bitwise_xor %2, %2 : (tensor<28x96x45xi1>, tensor<28x96x45xi1>) -> tensor<28x96x45xi1>
    %4 = tosa.logical_not %2 : (tensor<28x96x45xi1>) -> tensor<28x96x45xi1>
    %5 = tosa.bitwise_or %4, %4 : (tensor<28x96x45xi1>, tensor<28x96x45xi1>) -> tensor<28x96x45xi1>
    %6 = tosa.reduce_any %5 {axis = 2 : i32} : (tensor<28x96x45xi1>) -> tensor<28x96x1xi1>
    %7 = tosa.argmax %3 {axis = 1 : i32} : (tensor<28x96x45xi1>) -> tensor<28x45xi32>
    %8 = tosa.reduce_any %6 {axis = 0 : i32} : (tensor<28x96x1xi1>) -> tensor<1x96x1xi1>
    return %1, %7, %8 : tensor<100x224x116x87xf32>, tensor<28x45xi32>, tensor<1x96x1xi1>
  }
}
