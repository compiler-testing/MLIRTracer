module {
  func.func @main(%arg0: tensor<21x54x46xi1>) -> tensor<42x162x92xi1> {
    %t_0 = tosa.const_shape {values = dense<[ 2, 3, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.tile %arg0, %t_0 : (tensor<21x54x46xi1>, !tosa.shape<3>) -> tensor<42x162x92xi1>
    return %0 : tensor<42x162x92xi1>
  }
}
