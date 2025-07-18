module {
  func.func @main(%arg0: tensor<24xf32>, %arg1: tensor<33x19x26xi1>, %arg2: tensor<1x1x26xi1>) -> (tensor<4xf32>, tensor<24xf32>, tensor<22x741xi1>, tensor<1xi1>, tensor<24xf32>, tensor<24xf32>, tensor<1xf32>, tensor<33x19x1xi1>) {
    %0 = tosa.exp %arg0 : (tensor<24xf32>) -> tensor<24xf32>
    %1 = tosa.floor %0 : (tensor<24xf32>) -> tensor<24xf32>
    %2 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<24xf32>) -> tensor<1xf32>
    %3 = tosa.concat %2, %2 {axis = 0 : i32} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %4 = tosa.concat %3, %3 {axis = 0 : i32} : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
    %5 = tosa.logical_and %arg1, %arg2 : (tensor<33x19x26xi1>, tensor<1x1x26xi1>) -> tensor<33x19x26xi1>
    %6 = tosa.log %4 : (tensor<4xf32>) -> tensor<4xf32>
    %7 = tosa.bitwise_or %5, %5 : (tensor<33x19x26xi1>, tensor<33x19x26xi1>) -> tensor<33x19x26xi1>
    %8 = tosa.reduce_max %7 {axis = 2 : i32} : (tensor<33x19x26xi1>) -> tensor<33x19x1xi1>
    %9 = tosa.reciprocal %6 : (tensor<4xf32>) -> tensor<4xf32>
    %10 = tosa.logical_xor %8, %8 : (tensor<33x19x1xi1>, tensor<33x19x1xi1>) -> tensor<33x19x1xi1>
    %11 = tosa.log %0 : (tensor<24xf32>) -> tensor<24xf32>
    %12 = tosa.clz %5 : (tensor<33x19x26xi1>) -> tensor<33x19x26xi1>
    %13 = tosa.minimum %0, %1 : (tensor<24xf32>, tensor<24xf32>) -> tensor<24xf32>
    %14 = tosa.bitwise_xor %12, %12 : (tensor<33x19x26xi1>, tensor<33x19x26xi1>) -> tensor<33x19x26xi1>
    %r_15 = tosa.const_shape {values = dense<[ 22, 741 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %15 = tosa.reshape %14, %r_15 : (tensor<33x19x26xi1>, !tosa.shape<2>) -> tensor<22x741xi1>
    %16 = tosa.greater_equal %2, %2 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %17 = tosa.bitwise_or %10, %10 : (tensor<33x19x1xi1>, tensor<33x19x1xi1>) -> tensor<33x19x1xi1>
    %18 = tosa.reciprocal %11 : (tensor<24xf32>) -> tensor<24xf32>
    %19 = tosa.arithmetic_right_shift %17, %8 {round = true} : (tensor<33x19x1xi1>, tensor<33x19x1xi1>) -> tensor<33x19x1xi1>
    %20 = tosa.log %11 : (tensor<24xf32>) -> tensor<24xf32>
    %21 = tosa.exp %2 : (tensor<1xf32>) -> tensor<1xf32>
    %22 = tosa.reduce_max %19 {axis = 2 : i32} : (tensor<33x19x1xi1>) -> tensor<33x19x1xi1>
    return %9, %13, %15, %16, %18, %20, %21, %22 : tensor<4xf32>, tensor<24xf32>, tensor<22x741xi1>, tensor<1xi1>, tensor<24xf32>, tensor<24xf32>, tensor<1xf32>, tensor<33x19x1xi1>
  }
}
