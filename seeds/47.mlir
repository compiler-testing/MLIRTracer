module {
  func.func @main(%arg0: tensor<21x81x69x82x30x19xi16>, %arg1: tensor<1x1x69x1x1x19xi16>, %arg2: tensor<13x78x21x10xf32>) -> (tensor<21x81x69x82x30x19xi16>, tensor<3x156x42x10xf32>, tensor<1x78x21x10xf32>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<21x81x69x82x30x19xi16>, tensor<1x1x69x1x1x19xi16>) -> tensor<21x81x69x82x30x19xi16>
    %1 = tosa.reduce_product %arg2 {axis = 0 : i32} : (tensor<13x78x21x10xf32>) -> tensor<1x78x21x10xf32>
    %2 = tosa.reciprocal %1 : (tensor<1x78x21x10xf32>) -> tensor<1x78x21x10xf32>
    %3 = tosa.reverse %1 {axis = 3 : i32} : (tensor<1x78x21x10xf32>) -> tensor<1x78x21x10xf32>
    %4 = tosa.identity %2 : (tensor<1x78x21x10xf32>) -> tensor<1x78x21x10xf32>
    %5 = tosa.maximum %3, %4 : (tensor<1x78x21x10xf32>, tensor<1x78x21x10xf32>) -> tensor<1x78x21x10xf32>
    %6 = tosa.bitwise_not %0 : (tensor<21x81x69x82x30x19xi16>) -> tensor<21x81x69x82x30x19xi16>
    %7 = tosa.abs %4 : (tensor<1x78x21x10xf32>) -> tensor<1x78x21x10xf32>
    %t_8 = tosa.const_shape {values = dense<[ 3, 2, 2, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %8 = tosa.tile %5, %t_8 : (tensor<1x78x21x10xf32>, !tosa.shape<4>) -> tensor<3x156x42x10xf32>
    %9 = tosa.tanh %8 : (tensor<3x156x42x10xf32>) -> tensor<3x156x42x10xf32>
    %10 = tosa.sub %9, %9 : (tensor<3x156x42x10xf32>, tensor<3x156x42x10xf32>) -> tensor<3x156x42x10xf32>
    %11 = tosa.logical_left_shift %6, %6 : (tensor<21x81x69x82x30x19xi16>, tensor<21x81x69x82x30x19xi16>) -> tensor<21x81x69x82x30x19xi16>
    %in_zp_12 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_12 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %12 = tosa.negate %10, %in_zp_12, %out_zp_12 : (tensor<3x156x42x10xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3x156x42x10xf32>
    %13 = tosa.minimum %7, %1 : (tensor<1x78x21x10xf32>, tensor<1x78x21x10xf32>) -> tensor<1x78x21x10xf32>
    return %11, %12, %13 : tensor<21x81x69x82x30x19xi16>, tensor<3x156x42x10xf32>, tensor<1x78x21x10xf32>
  }
}
