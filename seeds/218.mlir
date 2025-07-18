module {
  func.func @main(%arg0: tensor<11x44x27x66xi1>, %arg1: tensor<65x26x34x83x63x29xf32>) -> (tensor<1x44x1x66xi1>, tensor<1x44x1x66xi1>, tensor<1x44x1x66xi1>, tensor<65x26x34x83x63x29xf32>, tensor<65x26x34x83x63x29xf32>, tensor<1x44x1x1xi1>) {
    %0 = tosa.reduce_any %arg0 {axis = 2 : i32} : (tensor<11x44x27x66xi1>) -> tensor<11x44x1x66xi1>
    %1 = tosa.reduce_any %0 {axis = 0 : i32} : (tensor<11x44x1x66xi1>) -> tensor<1x44x1x66xi1>
    %2 = tosa.identity %1 : (tensor<1x44x1x66xi1>) -> tensor<1x44x1x66xi1>
    %3 = tosa.floor %arg1 : (tensor<65x26x34x83x63x29xf32>) -> tensor<65x26x34x83x63x29xf32>
    %4 = tosa.logical_and %2, %1 : (tensor<1x44x1x66xi1>, tensor<1x44x1x66xi1>) -> tensor<1x44x1x66xi1>
    %5 = tosa.abs %3 : (tensor<65x26x34x83x63x29xf32>) -> tensor<65x26x34x83x63x29xf32>
    %6 = tosa.pow %5, %5 : (tensor<65x26x34x83x63x29xf32>, tensor<65x26x34x83x63x29xf32>) -> tensor<65x26x34x83x63x29xf32>
    %7 = tosa.pow %6, %6 : (tensor<65x26x34x83x63x29xf32>, tensor<65x26x34x83x63x29xf32>) -> tensor<65x26x34x83x63x29xf32>
    %8 = tosa.clz %2 : (tensor<1x44x1x66xi1>) -> tensor<1x44x1x66xi1>
    %9 = tosa.logical_left_shift %1, %2 : (tensor<1x44x1x66xi1>, tensor<1x44x1x66xi1>) -> tensor<1x44x1x66xi1>
    %10 = tosa.arithmetic_right_shift %1, %2 {round = false} : (tensor<1x44x1x66xi1>, tensor<1x44x1x66xi1>) -> tensor<1x44x1x66xi1>
    %11 = tosa.pow %5, %7 : (tensor<65x26x34x83x63x29xf32>, tensor<65x26x34x83x63x29xf32>) -> tensor<65x26x34x83x63x29xf32>
    %in_zp_12 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_12 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %12 = tosa.negate %7, %in_zp_12, %out_zp_12 : (tensor<65x26x34x83x63x29xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<65x26x34x83x63x29xf32>
    %13 = tosa.reduce_all %4 {axis = 3 : i32} : (tensor<1x44x1x66xi1>) -> tensor<1x44x1x1xi1>
    return %8, %9, %10, %11, %12, %13 : tensor<1x44x1x66xi1>, tensor<1x44x1x66xi1>, tensor<1x44x1x66xi1>, tensor<65x26x34x83x63x29xf32>, tensor<65x26x34x83x63x29xf32>, tensor<1x44x1x1xi1>
  }
}
