module {
  func.func @main(%arg0: tensor<11x8x45xf32>, %arg1: tensor<23x61x44x26xf32>, %arg2: tensor<52x10x95x10xf32>, %arg3: tensor<52xf32>, %arg4: tensor<94x42xi1>, %arg5: tensor<94x1xi1>) -> (tensor<23x133x141x1xf32>, tensor<94x42xi1>, tensor<11x8x1xf32>) {
    %0 = tosa.log %arg0 : (tensor<11x8x45xf32>) -> tensor<11x8x45xf32>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 2, 1, 2, 1>, stride = array<i64: 2, 1>, out_shape = array<i64: 23, 133, 141, 52>} : (tensor<23x61x44x26xf32>, tensor<52x10x95x10xf32>, tensor<52xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<23x133x141x52xf32>
    %2 = tosa.reduce_max %1 {axis = 3 : i32} : (tensor<23x133x141x52xf32>) -> tensor<23x133x141x1xf32>
    %3 = tosa.logical_or %arg4, %arg5 : (tensor<94x42xi1>, tensor<94x1xi1>) -> tensor<94x42xi1>
    %4 = tosa.reduce_product %0 {axis = 2 : i32} : (tensor<11x8x45xf32>) -> tensor<11x8x1xf32>
    return %2, %3, %4 : tensor<23x133x141x1xf32>, tensor<94x42xi1>, tensor<11x8x1xf32>
  }
}
