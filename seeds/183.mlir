module {
  func.func @main(%arg0: tensor<83x18x24x10x76x31xi32>, %arg1: tensor<1x1x1x10x76x31xi32>, %arg2: tensor<18x83x72xf32>, %arg3: tensor<71x24xi1>, %arg4: tensor<33x36x49x85xf32>) -> (tensor<83x72xi32>, tensor<71x1xi1>, tensor<83x18x24x10x76x31xi32>, tensor<33x36x49x85xi1>, tensor<33x36x49x85xf32>, tensor<33x36x49x85xf32>, tensor<33x108x98x85xi1>, tensor<33x36x49x85xi1>, tensor<33x36x49x85xf32>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<83x18x24x10x76x31xi32>, tensor<1x1x1x10x76x31xi32>) -> tensor<83x18x24x10x76x31xi32>
    %1 = tosa.identity %0 : (tensor<83x18x24x10x76x31xi32>) -> tensor<83x18x24x10x76x31xi32>
    %2 = tosa.argmax %arg2 {axis = 0 : i32} : (tensor<18x83x72xf32>) -> tensor<83x72xi32>
    %3 = tosa.reduce_any %arg3 {axis = 1 : i32} : (tensor<71x24xi1>) -> tensor<71x1xi1>
    %4 = tosa.log %arg4 : (tensor<33x36x49x85xf32>) -> tensor<33x36x49x85xf32>
    %5 = tosa.sub %1, %0 : (tensor<83x18x24x10x76x31xi32>, tensor<83x18x24x10x76x31xi32>) -> tensor<83x18x24x10x76x31xi32>
    %6 = tosa.greater %4, %4 : (tensor<33x36x49x85xf32>, tensor<33x36x49x85xf32>) -> tensor<33x36x49x85xi1>
    %7 = tosa.floor %4 : (tensor<33x36x49x85xf32>) -> tensor<33x36x49x85xf32>
    %8 = tosa.bitwise_and %6, %6 : (tensor<33x36x49x85xi1>, tensor<33x36x49x85xi1>) -> tensor<33x36x49x85xi1>
    %9 = tosa.maximum %4, %4 : (tensor<33x36x49x85xf32>, tensor<33x36x49x85xf32>) -> tensor<33x36x49x85xf32>
    %10 = tosa.maximum %4, %7 : (tensor<33x36x49x85xf32>, tensor<33x36x49x85xf32>) -> tensor<33x36x49x85xf32>
    %11 = tosa.bitwise_or %6, %6 : (tensor<33x36x49x85xi1>, tensor<33x36x49x85xi1>) -> tensor<33x36x49x85xi1>
    %12 = tosa.greater %4, %7 : (tensor<33x36x49x85xf32>, tensor<33x36x49x85xf32>) -> tensor<33x36x49x85xi1>
    %t_13 = tosa.const_shape {values = dense<[ 1, 3, 2, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %13 = tosa.tile %11, %t_13 : (tensor<33x36x49x85xi1>, !tosa.shape<4>) -> tensor<33x108x98x85xi1>
    %14 = tosa.logical_or %12, %6 : (tensor<33x36x49x85xi1>, tensor<33x36x49x85xi1>) -> tensor<33x36x49x85xi1>
    %15 = tosa.sigmoid %4 : (tensor<33x36x49x85xf32>) -> tensor<33x36x49x85xf32>
    return %2, %3, %5, %8, %9, %10, %13, %14, %15 : tensor<83x72xi32>, tensor<71x1xi1>, tensor<83x18x24x10x76x31xi32>, tensor<33x36x49x85xi1>, tensor<33x36x49x85xf32>, tensor<33x36x49x85xf32>, tensor<33x108x98x85xi1>, tensor<33x36x49x85xi1>, tensor<33x36x49x85xf32>
  }
}
