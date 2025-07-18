module {
  func.func @main(%arg0: tensor<37x42x73xf32>, %arg1: tensor<37x1x73xf32>, %arg2: tensor<88x42x21x64xi1>) -> (tensor<37x73x42xf32>, tensor<37x42x73xf32>, tensor<37x42x73xf32>, tensor<1x42x1x1xi1>, tensor<88x42x1xi32>, tensor<1x42x1x1xi1>, tensor<37x42x73xf32>, tensor<88x1x1xi32>, tensor<1x42x1x1xi1>, tensor<88x42x1x64xi1>, tensor<1x42x1x1xi1>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<37x42x73xf32>, tensor<37x1x73xf32>) -> tensor<37x42x73xf32>
    %1 = tosa.reduce_all %arg2 {axis = 2 : i32} : (tensor<88x42x21x64xi1>) -> tensor<88x42x1x64xi1>
    %2 = tosa.reduce_max %1 {axis = 3 : i32} : (tensor<88x42x1x64xi1>) -> tensor<88x42x1x1xi1>
    %in_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.negate %0, %in_zp_3, %out_zp_3 : (tensor<37x42x73xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<37x42x73xf32>
    %4 = tosa.abs %3 : (tensor<37x42x73xf32>) -> tensor<37x42x73xf32>
    %5 = tosa.argmax %2 {axis = 3 : i32} : (tensor<88x42x1x1xi1>) -> tensor<88x42x1xi32>
    %6 = tosa.minimum %4, %3 : (tensor<37x42x73xf32>, tensor<37x42x73xf32>) -> tensor<37x42x73xf32>
    %7 = tosa.tanh %6 : (tensor<37x42x73xf32>) -> tensor<37x42x73xf32>
    %8 = tosa.reciprocal %7 : (tensor<37x42x73xf32>) -> tensor<37x42x73xf32>
    %9 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %10 = tosa.transpose %8 {perms = array<i32: 0, 2, 1>} : (tensor<37x42x73xf32>) -> tensor<37x73x42xf32>
    %11 = tosa.logical_right_shift %5, %5 : (tensor<88x42x1xi32>, tensor<88x42x1xi32>) -> tensor<88x42x1xi32>
    %12 = tosa.pow %8, %4 : (tensor<37x42x73xf32>, tensor<37x42x73xf32>) -> tensor<37x42x73xf32>
    %13 = tosa.reduce_min %5 {axis = 1 : i32} : (tensor<88x42x1xi32>) -> tensor<88x1x1xi32>
    %14 = tosa.reduce_any %2 {axis = 0 : i32} : (tensor<88x42x1x1xi1>) -> tensor<1x42x1x1xi1>
    %15 = tosa.exp %4 : (tensor<37x42x73xf32>) -> tensor<37x42x73xf32>
    %16 = tosa.clz %13 : (tensor<88x1x1xi32>) -> tensor<88x1x1xi32>
    %17 = tosa.logical_or %14, %14 : (tensor<1x42x1x1xi1>, tensor<1x42x1x1xi1>) -> tensor<1x42x1x1xi1>
    %18 = tosa.bitwise_xor %16, %16 : (tensor<88x1x1xi32>, tensor<88x1x1xi32>) -> tensor<88x1x1xi32>
    %19 = tosa.reduce_all %17 {axis = 3 : i32} : (tensor<1x42x1x1xi1>) -> tensor<1x42x1x1xi1>
    %20 = tosa.minimum %11, %11 : (tensor<88x42x1xi32>, tensor<88x42x1xi32>) -> tensor<88x42x1xi32>
    %21 = tosa.logical_not %17 : (tensor<1x42x1x1xi1>) -> tensor<1x42x1x1xi1>
    %22 = tosa.logical_xor %21, %17 : (tensor<1x42x1x1xi1>, tensor<1x42x1x1xi1>) -> tensor<1x42x1x1xi1>
    %23 = tosa.rsqrt %8 : (tensor<37x42x73xf32>) -> tensor<37x42x73xf32>
    %in_zp_24 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_24 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %24 = tosa.negate %18, %in_zp_24, %out_zp_24 : (tensor<88x1x1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<88x1x1xi32>
    %25 = tosa.reverse %24 {axis = 2 : i32} : (tensor<88x1x1xi32>) -> tensor<88x1x1xi32>
    %26 = tosa.logical_not %14 : (tensor<1x42x1x1xi1>) -> tensor<1x42x1x1xi1>
    %27 = tosa.logical_not %1 : (tensor<88x42x1x64xi1>) -> tensor<88x42x1x64xi1>
    %28 = tosa.reduce_all %14 {axis = 0 : i32} : (tensor<1x42x1x1xi1>) -> tensor<1x42x1x1xi1>
    return %10, %12, %15, %19, %20, %22, %23, %25, %26, %27, %28 : tensor<37x73x42xf32>, tensor<37x42x73xf32>, tensor<37x42x73xf32>, tensor<1x42x1x1xi1>, tensor<88x42x1xi32>, tensor<1x42x1x1xi1>, tensor<37x42x73xf32>, tensor<88x1x1xi32>, tensor<1x42x1x1xi1>, tensor<88x42x1x64xi1>, tensor<1x42x1x1xi1>
  }
}
