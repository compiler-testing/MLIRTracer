module {
  func.func @main(%arg0: tensor<93x4x98x61xf32>, %arg1: tensor<10x96x11x69xf32>, %arg2: tensor<10xf32>, %arg3: tensor<i32>, %arg4: tensor<13xi1>, %arg5: tensor<13xi1>) -> (tensor<13xi1>, tensor<i32>, tensor<93x104x111x10xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 2>, stride = array<i64: 2, 1>, out_shape = array<i64: 93, 104, 111, 10>} : (tensor<93x4x98x61xf32>, tensor<10x96x11x69xf32>, tensor<10xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<93x104x111x10xf32>
    %1 = tosa.add %0, %0 : (tensor<93x104x111x10xf32>, tensor<93x104x111x10xf32>) -> tensor<93x104x111x10xf32>
    %2 = tosa.bitwise_not %arg3 : (tensor<i32>) -> tensor<i32>
    %3 = tosa.logical_and %arg4, %arg5 : (tensor<13xi1>, tensor<13xi1>) -> tensor<13xi1>
    %4 = tosa.sub %2, %2 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = tosa.greater %1, %0 : (tensor<93x104x111x10xf32>, tensor<93x104x111x10xf32>) -> tensor<93x104x111x10xi1>
    return %3, %4, %5 : tensor<13xi1>, tensor<i32>, tensor<93x104x111x10xi1>
  }
}
