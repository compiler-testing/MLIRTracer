module {
  func.func @main(%arg0: tensor<35x73x54x96x34xf32>, %arg1: tensor<73xi64>, %arg2: tensor<7x79x77xi1>, %arg3: tensor<1x79x77xi1>) -> (tensor<35x73x54x96x34xf32>, tensor<2xi64>, tensor<7x79x1xi1>, tensor<7x79x77xi1>, tensor<1xi64>) {
    %0 = tosa.reciprocal %arg0 : (tensor<35x73x54x96x34xf32>) -> tensor<35x73x54x96x34xf32>
    %1 = tosa.reduce_max %arg1 {axis = 0 : i32} : (tensor<73xi64>) -> tensor<1xi64>
    %t_2 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.tile %1, %t_2 : (tensor<1xi64>, !tosa.shape<1>) -> tensor<2xi64>
    %3 = tosa.logical_xor %arg2, %arg3 : (tensor<7x79x77xi1>, tensor<1x79x77xi1>) -> tensor<7x79x77xi1>
    %4 = tosa.logical_not %3 : (tensor<7x79x77xi1>) -> tensor<7x79x77xi1>
    %5 = tosa.maximum %1, %1 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %6 = tosa.reduce_product %3 {axis = 2 : i32} : (tensor<7x79x77xi1>) -> tensor<7x79x1xi1>
    %7 = tosa.bitwise_and %3, %4 : (tensor<7x79x77xi1>, tensor<7x79x77xi1>) -> tensor<7x79x77xi1>
    %8 = tosa.arithmetic_right_shift %5, %1 {round = false} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %9 = tosa.abs %8 : (tensor<1xi64>) -> tensor<1xi64>
    %10 = tosa.arithmetic_right_shift %9, %1 {round = false} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    return %0, %2, %6, %7, %10 : tensor<35x73x54x96x34xf32>, tensor<2xi64>, tensor<7x79x1xi1>, tensor<7x79x77xi1>, tensor<1xi64>
  }
}
