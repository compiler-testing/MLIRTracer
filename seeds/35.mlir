module {
  func.func @main(%arg0: tensor<46x18x38x8x100xf32>, %arg1: tensor<76x97x20x96x26xi1>, %arg2: tensor<76x1x20x1x26xi1>, %arg3: tensor<67xi1>) -> (tensor<76x97x20x96x52xi1>, tensor<i32>, tensor<46x18x38x8x100xi1>) {
    %0 = tosa.rsqrt %arg0 : (tensor<46x18x38x8x100xf32>) -> tensor<46x18x38x8x100xf32>
    %1 = tosa.logical_and %arg1, %arg2 : (tensor<76x97x20x96x26xi1>, tensor<76x1x20x1x26xi1>) -> tensor<76x97x20x96x26xi1>
    %2 = tosa.floor %0 : (tensor<46x18x38x8x100xf32>) -> tensor<46x18x38x8x100xf32>
    %3 = tosa.reduce_all %arg3 {axis = 0 : i32} : (tensor<67xi1>) -> tensor<1xi1>
    %4 = tosa.greater %2, %0 : (tensor<46x18x38x8x100xf32>, tensor<46x18x38x8x100xf32>) -> tensor<46x18x38x8x100xi1>
    %5 = tosa.concat %1, %1 {axis = 4 : i32} : (tensor<76x97x20x96x26xi1>, tensor<76x97x20x96x26xi1>) -> tensor<76x97x20x96x52xi1>
    %in_zp_6 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_6 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %6 = tosa.negate %5, %in_zp_6, %out_zp_6 : (tensor<76x97x20x96x52xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<76x97x20x96x52xi1>
    %7 = tosa.reduce_max %3 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %in_zp_8 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_8 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %8 = tosa.negate %4, %in_zp_8, %out_zp_8 : (tensor<46x18x38x8x100xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<46x18x38x8x100xi1>
    %9 = tosa.argmax %7 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<i32>
    %10 = tosa.bitwise_not %8 : (tensor<46x18x38x8x100xi1>) -> tensor<46x18x38x8x100xi1>
    return %6, %9, %10 : tensor<76x97x20x96x52xi1>, tensor<i32>, tensor<46x18x38x8x100xi1>
  }
}
