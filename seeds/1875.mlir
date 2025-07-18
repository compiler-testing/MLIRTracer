module {
  func.func @main(%arg0: tensor<39x63xi64>) -> tensor<78x63xi1> {
    %t_0 = tosa.const_shape {values = dense<[ 2, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.tile %arg0, %t_0 : (tensor<39x63xi64>, !tosa.shape<2>) -> tensor<78x63xi64>
    %1 = tosa.minimum %0, %0 : (tensor<78x63xi64>, tensor<78x63xi64>) -> tensor<78x63xi64>
    %2 = tosa.bitwise_or %1, %1 : (tensor<78x63xi64>, tensor<78x63xi64>) -> tensor<78x63xi64>
    %3 = tosa.greater_equal %2, %0 : (tensor<78x63xi64>, tensor<78x63xi64>) -> tensor<78x63xi1>
    %4 = tosa.add %3, %3 : (tensor<78x63xi1>, tensor<78x63xi1>) -> tensor<78x63xi1>
    return %4 : tensor<78x63xi1>
  }
}
