module {
  func.func @main(%arg0: tensor<95x76x19x57x76xi64>, %arg1: tensor<18x77x18xi1>) -> (tensor<95x76x19x57x76xi64>, tensor<1x77xi32>, tensor<36x77x18xi1>) {
    %0 = tosa.abs %arg0 : (tensor<95x76x19x57x76xi64>) -> tensor<95x76x19x57x76xi64>
    %t_1 = tosa.const_shape {values = dense<[ 2, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %arg1, %t_1 : (tensor<18x77x18xi1>, !tosa.shape<3>) -> tensor<36x77x18xi1>
    %2 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<36x77x18xi1>) -> tensor<1x77x18xi1>
    %3 = tosa.argmax %2 {axis = 2 : i32} : (tensor<1x77x18xi1>) -> tensor<1x77xi32>
    %4 = tosa.abs %1 : (tensor<36x77x18xi1>) -> tensor<36x77x18xi1>
    return %0, %3, %4 : tensor<95x76x19x57x76xi64>, tensor<1x77xi32>, tensor<36x77x18xi1>
  }
}
