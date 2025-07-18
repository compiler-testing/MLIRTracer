module {
  func.func @main(%arg0: tensor<25x27xf32>, %arg1: tensor<43x66x3x15xf32>, %arg2: tensor<100x90x93x23xf32>, %arg3: tensor<100xf32>) -> (tensor<25x27xf32>, tensor<43x223x97x1xf32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<25x27xf32>) -> tensor<25x27xf32>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 2, 1, 1>, stride = array<i64: 2, 1>, out_shape = array<i64: 43, 223, 97, 100>} : (tensor<43x66x3x15xf32>, tensor<100x90x93x23xf32>, tensor<100xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<43x223x97x100xf32>
    %2 = tosa.reduce_max %1 {axis = 3 : i32} : (tensor<43x223x97x100xf32>) -> tensor<43x223x97x1xf32>
    return %0, %2 : tensor<25x27xf32>, tensor<43x223x97x1xf32>
  }
}
