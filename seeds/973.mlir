module {
  func.func @main(%arg0: tensor<94x45xi1>, %arg1: tensor<85x45xi32>, %arg2: tensor<1x1xi32>) -> (tensor<94x1xi1>, tensor<85x45xi32>, tensor<85x1xi1>) {
    %0 = tosa.reduce_any %arg0 {axis = 1 : i32} : (tensor<94x45xi1>) -> tensor<94x1xi1>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<94x1xi1>) -> tensor<94x1xi1>
    %2 = tosa.maximum %arg1, %arg2 : (tensor<85x45xi32>, tensor<1x1xi32>) -> tensor<85x45xi32>
    %3 = tosa.logical_xor %1, %0 : (tensor<94x1xi1>, tensor<94x1xi1>) -> tensor<94x1xi1>
    %4 = tosa.bitwise_xor %3, %0 : (tensor<94x1xi1>, tensor<94x1xi1>) -> tensor<94x1xi1>
    %5 = tosa.maximum %2, %2 : (tensor<85x45xi32>, tensor<85x45xi32>) -> tensor<85x45xi32>
    %6 = tosa.logical_left_shift %2, %2 : (tensor<85x45xi32>, tensor<85x45xi32>) -> tensor<85x45xi32>
    %7 = tosa.equal %5, %5 : (tensor<85x45xi32>, tensor<85x45xi32>) -> tensor<85x45xi1>
    %8 = tosa.reduce_max %7 {axis = 1 : i32} : (tensor<85x45xi1>) -> tensor<85x1xi1>
    %9 = tosa.bitwise_not %8 : (tensor<85x1xi1>) -> tensor<85x1xi1>
    return %4, %6, %9 : tensor<94x1xi1>, tensor<85x45xi32>, tensor<85x1xi1>
  }
}
