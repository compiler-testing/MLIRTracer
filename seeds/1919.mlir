module {
  func.func @main(%arg0: tensor<82x96x46x64x34xi1>, %arg1: tensor<26x38x9xi64>, %arg2: tensor<26x1x1xi64>, %arg3: tensor<36x47xf32>) -> (tensor<4x12x7x1x2xi1>, tensor<36x47xf32>, tensor<26x38x9xi1>) {
    %0 = tosa.logical_not %arg0 : (tensor<82x96x46x64x34xi1>) -> tensor<82x96x46x64x34xi1>
    %1 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<82x96x46x64x34xi1>, tensor<82x96x46x64x34xi1>) -> tensor<164x96x46x64x34xi1>
    %2 = tosa.add %1, %1 : (tensor<164x96x46x64x34xi1>, tensor<164x96x46x64x34xi1>) -> tensor<164x96x46x64x34xi1>
    %3 = tosa.logical_left_shift %2, %1 : (tensor<164x96x46x64x34xi1>, tensor<164x96x46x64x34xi1>) -> tensor<164x96x46x64x34xi1>
    %4 = tosa.arithmetic_right_shift %3, %3 {round = true} : (tensor<164x96x46x64x34xi1>, tensor<164x96x46x64x34xi1>) -> tensor<164x96x46x64x34xi1>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %5 = tosa.negate %4, %in_zp_5, %out_zp_5 : (tensor<164x96x46x64x34xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<164x96x46x64x34xi1>
    %s_6_start = tosa.const_shape {values = dense<[ 24, 47, 39, 63, 6 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_6_size = tosa.const_shape {values = dense<[ 4, 12, 7, 1, 2 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %6 = tosa.slice %5, %s_6_start, %s_6_size : (tensor<164x96x46x64x34xi1>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<4x12x7x1x2xi1>
    %7 = tosa.abs %6 : (tensor<4x12x7x1x2xi1>) -> tensor<4x12x7x1x2xi1>
    %8 = tosa.logical_or %7, %6 : (tensor<4x12x7x1x2xi1>, tensor<4x12x7x1x2xi1>) -> tensor<4x12x7x1x2xi1>
    %9 = tosa.greater_equal %arg1, %arg2 : (tensor<26x38x9xi64>, tensor<26x1x1xi64>) -> tensor<26x38x9xi1>
    %10 = tosa.tanh %arg3 : (tensor<36x47xf32>) -> tensor<36x47xf32>
    %11 = tosa.logical_right_shift %9, %9 : (tensor<26x38x9xi1>, tensor<26x38x9xi1>) -> tensor<26x38x9xi1>
    %12 = tosa.arithmetic_right_shift %11, %9 {round = true} : (tensor<26x38x9xi1>, tensor<26x38x9xi1>) -> tensor<26x38x9xi1>
    %in_zp_13 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_13 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %13 = tosa.negate %12, %in_zp_13, %out_zp_13 : (tensor<26x38x9xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<26x38x9xi1>
    return %8, %10, %13 : tensor<4x12x7x1x2xi1>, tensor<36x47xf32>, tensor<26x38x9xi1>
  }
}
