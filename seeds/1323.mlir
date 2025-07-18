module {
  func.func @main(%arg0: tensor<77x47x64x93xi32>) -> tensor<231x47x192x93xi32> {
    %t_0 = tosa.const_shape {values = dense<[ 3, 1, 3, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.tile %arg0, %t_0 : (tensor<77x47x64x93xi32>, !tosa.shape<4>) -> tensor<231x47x192x93xi32>
    return %0 : tensor<231x47x192x93xi32>
  }
}
