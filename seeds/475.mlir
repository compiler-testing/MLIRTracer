module {
  func.func @main(%arg0: tensor<75x88x99xi1>, %arg1: tensor<75x88x99xi1>, %arg2: tensor<36xf32>, %arg3: tensor<36xf32>, %arg4: tensor<i32>, %arg5: tensor<i32>) -> (tensor<225x176x198xi1>, tensor<36xf32>, tensor<i32>, tensor<36xf32>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<75x88x99xi1>, tensor<75x88x99xi1>) -> tensor<75x88x99xi1>
    %1 = tosa.add %0, %0 : (tensor<75x88x99xi1>, tensor<75x88x99xi1>) -> tensor<75x88x99xi1>
    %t_2 = tosa.const_shape {values = dense<[ 3, 2, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.tile %1, %t_2 : (tensor<75x88x99xi1>, !tosa.shape<3>) -> tensor<225x176x198xi1>
    %3 = tosa.bitwise_not %2 : (tensor<225x176x198xi1>) -> tensor<225x176x198xi1>
    %4 = tosa.sub %3, %2 : (tensor<225x176x198xi1>, tensor<225x176x198xi1>) -> tensor<225x176x198xi1>
    %5 = tosa.pow %arg2, %arg3 : (tensor<36xf32>, tensor<36xf32>) -> tensor<36xf32>
    %6 = tosa.bitwise_and %4, %2 : (tensor<225x176x198xi1>, tensor<225x176x198xi1>) -> tensor<225x176x198xi1>
    %7 = tosa.logical_and %6, %6 : (tensor<225x176x198xi1>, tensor<225x176x198xi1>) -> tensor<225x176x198xi1>
    %8 = tosa.arithmetic_right_shift %7, %4 {round = false} : (tensor<225x176x198xi1>, tensor<225x176x198xi1>) -> tensor<225x176x198xi1>
    %9 = tosa.intdiv %arg4, %arg5 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %10 = tosa.maximum %5, %5 : (tensor<36xf32>, tensor<36xf32>) -> tensor<36xf32>
    %11 = tosa.arithmetic_right_shift %9, %9 {round = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %12 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %13 = tosa.transpose %5 {perms = array<i32: 0>} : (tensor<36xf32>) -> tensor<36xf32>
    return %8, %10, %11, %13 : tensor<225x176x198xi1>, tensor<36xf32>, tensor<i32>, tensor<36xf32>
  }
}
