module {
  func.func @main(%arg0: tensor<69x52x7x22xf32>, %arg1: tensor<30xi1>) -> (tensor<69x52x7x22xf32>, tensor<4xi1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<69x52x7x22xf32>) -> tensor<69x52x7x22xf32>
    %1 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<30xi1>) -> tensor<1xi1>
    %2 = tosa.abs %0 : (tensor<69x52x7x22xf32>) -> tensor<69x52x7x22xf32>
    %s_3_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_3_size = tosa.const_shape {values = dense<[ 4 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.slice %1, %s_3_start, %s_3_size : (tensor<1xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<4xi1>
    %4 = tosa.bitwise_and %3, %3 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %2, %4 : tensor<69x52x7x22xf32>, tensor<4xi1>
  }
}
