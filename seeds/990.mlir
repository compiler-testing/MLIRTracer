module {
  func.func @main(%arg0: tensor<88x36xi16>, %arg1: tensor<29x73xi1>) -> (tensor<1xi1>, tensor<73x1x1x1xi1>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<88x36xi16>) -> tensor<36xi32>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<36xi32>) -> tensor<36xi32>
    %2 = tosa.reduce_product %1 {axis = 0 : i32} : (tensor<36xi32>) -> tensor<1xi32>
    %3 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<29x73xi1>) -> tensor<1x73xi1>
    %4 = tosa.greater %2, %2 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = tosa.logical_not %3 : (tensor<1x73xi1>) -> tensor<1x73xi1>
    %r_6 = tosa.const_shape {values = dense<[ 73, 1, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %6 = tosa.reshape %5, %r_6 : (tensor<1x73xi1>, !tosa.shape<4>) -> tensor<73x1x1x1xi1>
    %7 = tosa.reduce_max %6 {axis = 1 : i32} : (tensor<73x1x1x1xi1>) -> tensor<73x1x1x1xi1>
    return %4, %7 : tensor<1xi1>, tensor<73x1x1x1xi1>
  }
}
