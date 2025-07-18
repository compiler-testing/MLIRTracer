module {
  func.func @main(%arg0: tensor<13x16x82x73x5xf32>, %arg1: tensor<5x11x18x20xf32>, %arg2: tensor<94x94x100x86xf32>, %arg3: tensor<94xf32>, %arg4: tensor<i1>) -> (tensor<13x16x82x73x5xf32>, tensor<5x108x120x94xf32>, tensor<i1>) {
    %0 = tosa.reciprocal %arg0 : (tensor<13x16x82x73x5xf32>) -> tensor<13x16x82x73x5xf32>
    %1 = tosa.rsqrt %0 : (tensor<13x16x82x73x5xf32>) -> tensor<13x16x82x73x5xf32>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 5, 108, 120, 94>} : (tensor<5x11x18x20xf32>, tensor<94x94x100x86xf32>, tensor<94xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x108x120x94xf32>
    %3 = tosa.reciprocal %1 : (tensor<13x16x82x73x5xf32>) -> tensor<13x16x82x73x5xf32>
    %4 = tosa.reverse %2 {axis = 3 : i32} : (tensor<5x108x120x94xf32>) -> tensor<5x108x120x94xf32>
    %5 = tosa.bitwise_not %arg4 : (tensor<i1>) -> tensor<i1>
    return %3, %4, %5 : tensor<13x16x82x73x5xf32>, tensor<5x108x120x94xf32>, tensor<i1>
  }
}
