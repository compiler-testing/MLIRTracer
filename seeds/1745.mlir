module {
  func.func @main(%arg0: tensor<91x33x97x24x80xi32>, %arg1: tensor<91x1x97x1x1xi32>, %arg2: tensor<100x66x65xi32>) -> (tensor<91x33x97x24x80xi1>, tensor<1x66x65xi32>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<91x33x97x24x80xi32>, tensor<91x1x97x1x1xi32>) -> tensor<91x33x97x24x80xi1>
    %1 = tosa.reduce_max %arg2 {axis = 0 : i32} : (tensor<100x66x65xi32>) -> tensor<1x66x65xi32>
    return %0, %1 : tensor<91x33x97x24x80xi1>, tensor<1x66x65xi32>
  }
}
