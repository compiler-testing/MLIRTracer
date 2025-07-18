module {
  func.func @main(%arg0: tensor<38x66x65xf32>, %arg1: tensor<48x81x39x3xf32>, %arg2: tensor<23x83x61x54xf32>, %arg3: tensor<23xf32>, %arg4: tensor<71x74x17x90xi1>, %arg5: tensor<71x74x17x90xi1>) -> (tensor<71x74x17x90xi1>, tensor<1x66x65xf32>, tensor<48x245x140x23xf32>) {
    %0 = tosa.reciprocal %arg0 : (tensor<38x66x65xf32>) -> tensor<38x66x65xf32>
    %1 = tosa.rsqrt %0 : (tensor<38x66x65xf32>) -> tensor<38x66x65xf32>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 48, 245, 140, 23>} : (tensor<48x81x39x3xf32>, tensor<23x83x61x54xf32>, tensor<23xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<48x245x140x23xf32>
    %3 = tosa.clamp %2 {min_val = 1.200000e+01 : f32, max_val = 5.800000e+01 : f32} : (tensor<48x245x140x23xf32>) -> tensor<48x245x140x23xf32>
    %4 = tosa.reverse %3 {axis = 2 : i32} : (tensor<48x245x140x23xf32>) -> tensor<48x245x140x23xf32>
    %5 = tosa.logical_or %arg4, %arg5 : (tensor<71x74x17x90xi1>, tensor<71x74x17x90xi1>) -> tensor<71x74x17x90xi1>
    %6 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<38x66x65xf32>) -> tensor<1x66x65xf32>
    %7 = tosa.rsqrt %6 : (tensor<1x66x65xf32>) -> tensor<1x66x65xf32>
    %8 = tosa.minimum %4, %3 : (tensor<48x245x140x23xf32>, tensor<48x245x140x23xf32>) -> tensor<48x245x140x23xf32>
    return %5, %7, %8 : tensor<71x74x17x90xi1>, tensor<1x66x65xf32>, tensor<48x245x140x23xf32>
  }
}
