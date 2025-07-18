module {
  func.func @main(%arg0: tensor<69xi1>, %arg1: tensor<28x31x20x80x88xi8>, %arg2: tensor<1x1x20x1x88xi8>) -> (tensor<28x31x20x80x88xi1>, tensor<1xi1>) {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<69xi1>) -> tensor<1xi1>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %2 = tosa.sub %1, %0 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %t_3 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.tile %2, %t_3 : (tensor<1xi1>, !tosa.shape<1>) -> tensor<1xi1>
    %4 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.greater_equal %arg1, %arg2 : (tensor<28x31x20x80x88xi8>, tensor<1x1x20x1x88xi8>) -> tensor<28x31x20x80x88xi1>
    %6 = tosa.reduce_any %4 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %5, %6 : tensor<28x31x20x80x88xi1>, tensor<1xi1>
  }
}
