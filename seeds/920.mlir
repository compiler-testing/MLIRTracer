module {
  func.func @main(%arg0: tensor<55x64x95x64xf32>, %arg1: tensor<38x81x53x78xf32>, %arg2: tensor<38xf32>, %arg3: tensor<96x46x57x66xf32>, %arg4: tensor<96xf32>, %arg5: tensor<82xi1>) -> (tensor<195x545x96xi32>, tensor<1xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 1, 1, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 55, 147, 244, 38>} : (tensor<55x64x95x64xf32>, tensor<38x81x53x78xf32>, tensor<38xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<55x147x244x38xf32>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %0, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 2, 1, 1>, stride = array<i64: 1, 2>, out_shape = array<i64: 55, 195, 545, 96>} : (tensor<55x147x244x38xf32>, tensor<96x46x57x66xf32>, tensor<96xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<55x195x545x96xf32>
    %2 = tosa.reduce_all %arg5 {axis = 0 : i32} : (tensor<82xi1>) -> tensor<1xi1>
    %3 = tosa.argmax %1 {axis = 0 : i32} : (tensor<55x195x545x96xf32>) -> tensor<195x545x96xi32>
    %4 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %3, %4 : tensor<195x545x96xi32>, tensor<1xi1>
  }
}
