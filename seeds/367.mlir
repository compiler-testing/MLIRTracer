module {
  func.func @main(%arg0: tensor<54xi1>, %arg1: tensor<71xi8>, %arg2: tensor<1xi8>) -> (tensor<2xi1>, tensor<71xi1>) {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<54xi1>) -> tensor<1xi1>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %2 = tosa.logical_left_shift %1, %0 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %t_3 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.tile %2, %t_3 : (tensor<1xi1>, !tosa.shape<1>) -> tensor<2xi1>
    %4 = tosa.greater_equal %arg1, %arg2 : (tensor<71xi8>, tensor<1xi8>) -> tensor<71xi1>
    return %3, %4 : tensor<2xi1>, tensor<71xi1>
  }
}
