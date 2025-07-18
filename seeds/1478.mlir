module {
  func.func @main(%arg0: tensor<4x74xi64>, %arg1: tensor<4x13x7x48x88x42xi1>, %arg2: tensor<1x13x1x48x1x1xi1>, %arg3: tensor<95x29xf32>) -> (tensor<88x48x13x7x4x42xi1>, tensor<1x74xi1>, tensor<4x13x14x48x88x42xi1>, tensor<95x29xf32>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<4x74xi64>) -> tensor<4x74xi64>
    %1 = tosa.logical_xor %arg1, %arg2 : (tensor<4x13x7x48x88x42xi1>, tensor<1x13x1x48x1x1xi1>) -> tensor<4x13x7x48x88x42xi1>
    %2 = tosa.ceil %arg3 : (tensor<95x29xf32>) -> tensor<95x29xf32>
    %3 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
    %4 = tosa.transpose %1 {perms = array<i32: 4, 3, 1, 2, 0, 5>} : (tensor<4x13x7x48x88x42xi1>) -> tensor<88x48x13x7x4x42xi1>
    %5 = tosa.concat %1, %1 {axis = 2 : i32} : (tensor<4x13x7x48x88x42xi1>, tensor<4x13x7x48x88x42xi1>) -> tensor<4x13x14x48x88x42xi1>
    %6 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<4x74xi64>) -> tensor<1x74xi64>
    %7 = tosa.clamp %6 {min_val = 10 : i64, max_val = 74 : i64} : (tensor<1x74xi64>) -> tensor<1x74xi64>
    %8 = tosa.greater %7, %6 : (tensor<1x74xi64>, tensor<1x74xi64>) -> tensor<1x74xi1>
    %9 = tosa.bitwise_or %5, %5 : (tensor<4x13x14x48x88x42xi1>, tensor<4x13x14x48x88x42xi1>) -> tensor<4x13x14x48x88x42xi1>
    %in_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %10 = tosa.negate %8, %in_zp_10, %out_zp_10 : (tensor<1x74xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<1x74xi1>
    %11 = tosa.reduce_min %10 {axis = 0 : i32} : (tensor<1x74xi1>) -> tensor<1x74xi1>
    %12 = tosa.bitwise_and %9, %5 : (tensor<4x13x14x48x88x42xi1>, tensor<4x13x14x48x88x42xi1>) -> tensor<4x13x14x48x88x42xi1>
    %13 = tosa.floor %2 : (tensor<95x29xf32>) -> tensor<95x29xf32>
    return %4, %11, %12, %13 : tensor<88x48x13x7x4x42xi1>, tensor<1x74xi1>, tensor<4x13x14x48x88x42xi1>, tensor<95x29xf32>
  }
}
