module {
  func.func @main(%arg0: tensor<84x69xi32>) -> tensor<252x69xi32> {
    %t_0 = tosa.const_shape {values = dense<[ 3, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.tile %arg0, %t_0 : (tensor<84x69xi32>, !tosa.shape<2>) -> tensor<252x69xi32>
    return %0 : tensor<252x69xi32>
  }
}
