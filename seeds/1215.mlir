module {
  func.func @main(%arg0: tensor<100x68xf32>) -> (tensor<100x68xi1>, tensor<100x68xf32>, tensor<100x68xf32>, tensor<2xi32>, tensor<100x68xf32>, tensor<100x68xf32>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<100x68xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<100x68xf32>
    %1 = tosa.identity %0 : (tensor<100x68xf32>) -> tensor<100x68xf32>
    %2 = tosa.greater %1, %0 : (tensor<100x68xf32>, tensor<100x68xf32>) -> tensor<100x68xi1>
    %3 = tosa.identity %2 : (tensor<100x68xi1>) -> tensor<100x68xi1>
    %4 = tosa.greater_equal %1, %1 : (tensor<100x68xf32>, tensor<100x68xf32>) -> tensor<100x68xi1>
    %5 = tosa.reduce_sum %3 {axis = 1 : i32} : (tensor<100x68xi1>) -> tensor<100x1xi1>
    %6 = tosa.floor %0 : (tensor<100x68xf32>) -> tensor<100x68xf32>
    %7 = tosa.add %5, %5 : (tensor<100x1xi1>, tensor<100x1xi1>) -> tensor<100x1xi1>
    %8 = tosa.reciprocal %0 : (tensor<100x68xf32>) -> tensor<100x68xf32>
    %9 = tosa.pow %6, %8 : (tensor<100x68xf32>, tensor<100x68xf32>) -> tensor<100x68xf32>
    %10 = tosa.logical_and %7, %5 : (tensor<100x1xi1>, tensor<100x1xi1>) -> tensor<100x1xi1>
    %11 = tosa.abs %10 : (tensor<100x1xi1>) -> tensor<100x1xi1>
    %12 = tosa.logical_not %11 : (tensor<100x1xi1>) -> tensor<100x1xi1>
    %13 = tosa.logical_left_shift %12, %11 : (tensor<100x1xi1>, tensor<100x1xi1>) -> tensor<100x1xi1>
    %14 = tosa.rsqrt %8 : (tensor<100x68xf32>) -> tensor<100x68xf32>
    %15 = tosa.logical_left_shift %13, %13 : (tensor<100x1xi1>, tensor<100x1xi1>) -> tensor<100x1xi1>
    %16 = tosa.clz %15 : (tensor<100x1xi1>) -> tensor<100x1xi1>
    %17 = tosa.add %16, %12 : (tensor<100x1xi1>, tensor<100x1xi1>) -> tensor<100x1xi1>
    %18 = tosa.reduce_product %17 {axis = 1 : i32} : (tensor<100x1xi1>) -> tensor<100x1xi1>
    %19 = tosa.argmax %18 {axis = 0 : i32} : (tensor<100x1xi1>) -> tensor<1xi32>
    %20 = tosa.concat %19, %19 {axis = 0 : i32} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %21 = tosa.exp %0 : (tensor<100x68xf32>) -> tensor<100x68xf32>
    %22 = tosa.abs %20 : (tensor<2xi32>) -> tensor<2xi32>
    %23 = tosa.pow %6, %0 : (tensor<100x68xf32>, tensor<100x68xf32>) -> tensor<100x68xf32>
    %24 = tosa.exp %21 : (tensor<100x68xf32>) -> tensor<100x68xf32>
    return %4, %9, %14, %22, %23, %24 : tensor<100x68xi1>, tensor<100x68xf32>, tensor<100x68xf32>, tensor<2xi32>, tensor<100x68xf32>, tensor<100x68xf32>
  }
}
