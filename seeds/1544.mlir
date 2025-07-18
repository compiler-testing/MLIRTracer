module {
  func.func @main(%arg0: tensor<12x56x93x95xf32>, %arg1: tensor<76x5x22x20xf32>, %arg2: tensor<76xf32>, %arg3: tensor<i1>, %arg4: tensor<i1>) -> (tensor<i1>, tensor<12x117x117x76xf32>, tensor<12x117x117x76xf32>, tensor<12x76x117x117xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 2>, stride = array<i64: 2, 1>, out_shape = array<i64: 12, 117, 117, 76>} : (tensor<12x56x93x95xf32>, tensor<76x5x22x20xf32>, tensor<76xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x117x117x76xf32>
    %1 = tosa.maximum %0, %0 : (tensor<12x117x117x76xf32>, tensor<12x117x117x76xf32>) -> tensor<12x117x117x76xf32>
    %2 = tosa.rsqrt %1 : (tensor<12x117x117x76xf32>) -> tensor<12x117x117x76xf32>
    %3 = tosa.logical_or %arg3, %arg4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.minimum %2, %1 : (tensor<12x117x117x76xf32>, tensor<12x117x117x76xf32>) -> tensor<12x117x117x76xf32>
    %5 = tosa.identity %3 : (tensor<i1>) -> tensor<i1>
    %6 = tosa.greater_equal %4, %4 : (tensor<12x117x117x76xf32>, tensor<12x117x117x76xf32>) -> tensor<12x117x117x76xi1>
    %7 = tosa.log %0 : (tensor<12x117x117x76xf32>) -> tensor<12x117x117x76xf32>
    %8 = tosa.logical_left_shift %6, %6 : (tensor<12x117x117x76xi1>, tensor<12x117x117x76xi1>) -> tensor<12x117x117x76xi1>
    %9 = tosa.bitwise_and %8, %6 : (tensor<12x117x117x76xi1>, tensor<12x117x117x76xi1>) -> tensor<12x117x117x76xi1>
    %10 = tosa.minimum %0, %7 : (tensor<12x117x117x76xf32>, tensor<12x117x117x76xf32>) -> tensor<12x117x117x76xf32>
    %11 = tosa.rsqrt %0 : (tensor<12x117x117x76xf32>) -> tensor<12x117x117x76xf32>
    %12 = tosa.bitwise_or %5, %5 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %13 = tosa.tanh %11 : (tensor<12x117x117x76xf32>) -> tensor<12x117x117x76xf32>
    %14 = tosa.rsqrt %10 : (tensor<12x117x117x76xf32>) -> tensor<12x117x117x76xf32>
    %15 = "tosa.const"() {values = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %16 = tosa.transpose %9 {perms = array<i32: 0, 3, 1, 2>} : (tensor<12x117x117x76xi1>) -> tensor<12x76x117x117xi1>
    return %12, %13, %14, %16 : tensor<i1>, tensor<12x117x117x76xf32>, tensor<12x117x117x76xf32>, tensor<12x76x117x117xi1>
  }
}
