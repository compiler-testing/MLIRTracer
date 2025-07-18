module {
  func.func @main(%arg0: tensor<31xi1>, %arg1: tensor<76x39xf32>) -> (tensor<1xi1>, tensor<76x39xf32>) {
    %t_0 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.tile %arg0, %t_0 : (tensor<31xi1>, !tosa.shape<1>) -> tensor<62xi1>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<62xi1>) -> tensor<1xi1>
    %2 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %3 = tosa.reduce_all %2 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %4 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.logical_left_shift %4, %4 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.ceil %arg1 : (tensor<76x39xf32>) -> tensor<76x39xf32>
    return %5, %6 : tensor<1xi1>, tensor<76x39xf32>
  }
}
