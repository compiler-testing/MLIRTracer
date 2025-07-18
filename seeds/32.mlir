module {
  func.func @main(%arg0: tensor<6x53xi32>, %arg1: tensor<77xf32>) -> (tensor<6x53xi32>, tensor<i32>, tensor<1xf32>, tensor<6x53xi1>, tensor<3xf32>) {
    %0 = tosa.identity %arg0 : (tensor<6x53xi32>) -> tensor<6x53xi32>
    %1 = tosa.exp %arg1 : (tensor<77xf32>) -> tensor<77xf32>
    %2 = tosa.identity %0 : (tensor<6x53xi32>) -> tensor<6x53xi32>
    %3 = tosa.greater_equal %2, %0 : (tensor<6x53xi32>, tensor<6x53xi32>) -> tensor<6x53xi1>
    %4 = tosa.ceil %1 : (tensor<77xf32>) -> tensor<77xf32>
    %5 = tosa.bitwise_not %2 : (tensor<6x53xi32>) -> tensor<6x53xi32>
    %6 = tosa.maximum %1, %4 : (tensor<77xf32>, tensor<77xf32>) -> tensor<77xf32>
    %7 = tosa.pow %6, %1 : (tensor<77xf32>, tensor<77xf32>) -> tensor<77xf32>
    %8 = tosa.argmax %4 {axis = 0 : i32} : (tensor<77xf32>) -> tensor<i32>
    %9 = tosa.reduce_max %7 {axis = 0 : i32} : (tensor<77xf32>) -> tensor<1xf32>
    %10 = tosa.logical_and %3, %3 : (tensor<6x53xi1>, tensor<6x53xi1>) -> tensor<6x53xi1>
    %s_11_start = tosa.const_shape {values = dense<[ 26 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_11_size = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %11 = tosa.slice %7, %s_11_start, %s_11_size : (tensor<77xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<3xf32>
    return %5, %8, %9, %10, %11 : tensor<6x53xi32>, tensor<i32>, tensor<1xf32>, tensor<6x53xi1>, tensor<3xf32>
  }
}
