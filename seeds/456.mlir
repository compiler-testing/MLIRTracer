module {
  func.func @main(%arg0: tensor<52x36xf32>) -> tensor<156x108xf32> {
    %t_0 = tosa.const_shape {values = dense<[ 3, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.tile %arg0, %t_0 : (tensor<52x36xf32>, !tosa.shape<2>) -> tensor<156x108xf32>
    return %0 : tensor<156x108xf32>
  }
}
