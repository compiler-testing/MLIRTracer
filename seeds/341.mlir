module {
  func.func @main(%arg0: tensor<79x5x32x59x64x82xi1>, %arg1: tensor<79x5x32x59x1x1xi1>, %arg2: tensor<96xf32>) -> (tensor<79x5x32x59x64x82xi1>, tensor<1xf32>, tensor<1xi1>, tensor<1xf32>, tensor<1xf32>, tensor<2xi1>, tensor<1xi1>, tensor<1xf32>, tensor<1xi1>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<79x5x32x59x64x82xi1>, tensor<79x5x32x59x1x1xi1>) -> tensor<79x5x32x59x64x82xi1>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<79x5x32x59x64x82xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<79x5x32x59x64x82xi1>
    %2 = tosa.reduce_max %arg2 {axis = 0 : i32} : (tensor<96xf32>) -> tensor<1xf32>
    %3 = tosa.pow %2, %2 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %4 = tosa.reduce_max %3 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %5 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %6 = tosa.greater_equal %5, %2 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %in_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %7 = tosa.negate %6, %in_zp_7, %out_zp_7 : (tensor<1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %8 = tosa.reverse %4 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %9 = tosa.bitwise_or %7, %7 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %10 = tosa.bitwise_not %7 : (tensor<1xi1>) -> tensor<1xi1>
    %11 = tosa.reduce_product %9 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %12 = tosa.reduce_product %10 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %13 = tosa.maximum %3, %3 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %14 = tosa.sigmoid %5 : (tensor<1xf32>) -> tensor<1xf32>
    %15 = tosa.maximum %2, %13 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %16 = tosa.greater %15, %14 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %17 = tosa.logical_or %11, %6 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %18 = tosa.maximum %4, %15 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %19 = tosa.add %12, %12 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %20 = tosa.reduce_all %17 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %21 = tosa.maximum %13, %5 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %22 = tosa.ceil %13 : (tensor<1xf32>) -> tensor<1xf32>
    %23 = tosa.concat %20, %9 {axis = 0 : i32} : (tensor<1xi1>, tensor<1xi1>) -> tensor<2xi1>
    %24 = tosa.logical_or %23, %23 : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    %25 = tosa.abs %19 : (tensor<1xi1>) -> tensor<1xi1>
    %26 = tosa.logical_or %25, %25 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %27 = tosa.ceil %3 : (tensor<1xf32>) -> tensor<1xf32>
    %28 = tosa.equal %22, %13 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    return %1, %8, %16, %18, %21, %24, %26, %27, %28 : tensor<79x5x32x59x64x82xi1>, tensor<1xf32>, tensor<1xi1>, tensor<1xf32>, tensor<1xf32>, tensor<2xi1>, tensor<1xi1>, tensor<1xf32>, tensor<1xi1>
  }
}
