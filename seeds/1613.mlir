module {
  func.func @main(%arg0: tensor<58x65x89xf32>, %arg1: tensor<22x46x54x15xi16>, %arg2: tensor<22x1x54x15xi16>) -> (tensor<58x65x89xf32>, tensor<58x65x89xf32>, tensor<58x65xi1>, tensor<9x8xi1>, tensor<58x65x89xf32>, tensor<22x1x54x1xi16>) {
    %0 = tosa.sigmoid %arg0 : (tensor<58x65x89xf32>) -> tensor<58x65x89xf32>
    %1 = tosa.logical_right_shift %arg1, %arg2 : (tensor<22x46x54x15xi16>, tensor<22x1x54x15xi16>) -> tensor<22x46x54x15xi16>
    %2 = tosa.reduce_product %1 {axis = 1 : i32} : (tensor<22x46x54x15xi16>) -> tensor<22x1x54x15xi16>
    %3 = tosa.abs %0 : (tensor<58x65x89xf32>) -> tensor<58x65x89xf32>
    %4 = tosa.sigmoid %3 : (tensor<58x65x89xf32>) -> tensor<58x65x89xf32>
    %5 = tosa.argmax %4 {axis = 2 : i32} : (tensor<58x65x89xf32>) -> tensor<58x65xi32>
    %6 = tosa.intdiv %5, %5 : (tensor<58x65xi32>, tensor<58x65xi32>) -> tensor<58x65xi32>
    %7 = tosa.intdiv %6, %6 : (tensor<58x65xi32>, tensor<58x65xi32>) -> tensor<58x65xi32>
    %8 = tosa.equal %7, %5 : (tensor<58x65xi32>, tensor<58x65xi32>) -> tensor<58x65xi1>
    %9 = tosa.reduce_any %8 {axis = 0 : i32} : (tensor<58x65xi1>) -> tensor<1x65xi1>
    %10 = tosa.reduce_max %2 {axis = 3 : i32} : (tensor<22x1x54x15xi16>) -> tensor<22x1x54x1xi16>
    %11 = tosa.reduce_sum %9 {axis = 0 : i32} : (tensor<1x65xi1>) -> tensor<1x65xi1>
    %12 = tosa.exp %4 : (tensor<58x65x89xf32>) -> tensor<58x65x89xf32>
    %13 = tosa.reverse %10 {axis = 3 : i32} : (tensor<22x1x54x1xi16>) -> tensor<22x1x54x1xi16>
    %14 = tosa.reduce_any %11 {axis = 0 : i32} : (tensor<1x65xi1>) -> tensor<1x65xi1>
    %15 = tosa.reciprocal %4 : (tensor<58x65x89xf32>) -> tensor<58x65x89xf32>
    %s_16_start = tosa.const_shape {values = dense<[ 0, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_16_size = tosa.const_shape {values = dense<[ 9, 8 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %16 = tosa.slice %14, %s_16_start, %s_16_size : (tensor<1x65xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<9x8xi1>
    %17 = tosa.bitwise_not %16 : (tensor<9x8xi1>) -> tensor<9x8xi1>
    %18 = tosa.greater_equal %5, %6 : (tensor<58x65xi32>, tensor<58x65xi32>) -> tensor<58x65xi1>
    %19 = tosa.sub %17, %17 : (tensor<9x8xi1>, tensor<9x8xi1>) -> tensor<9x8xi1>
    %20 = tosa.tanh %3 : (tensor<58x65x89xf32>) -> tensor<58x65x89xf32>
    %21 = tosa.reduce_product %13 {axis = 3 : i32} : (tensor<22x1x54x1xi16>) -> tensor<22x1x54x1xi16>
    return %12, %15, %18, %19, %20, %21 : tensor<58x65x89xf32>, tensor<58x65x89xf32>, tensor<58x65xi1>, tensor<9x8xi1>, tensor<58x65x89xf32>, tensor<22x1x54x1xi16>
  }
}
