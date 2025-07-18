module {
  func.func @main(%arg0: tensor<91x27x61xf32>) -> tensor<273x27x183xi1> {
    %0 = tosa.sigmoid %arg0 : (tensor<91x27x61xf32>) -> tensor<91x27x61xf32>
    %t_1 = tosa.const_shape {values = dense<[ 3, 1, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %0, %t_1 : (tensor<91x27x61xf32>, !tosa.shape<3>) -> tensor<273x27x183xf32>
    %2 = tosa.abs %1 : (tensor<273x27x183xf32>) -> tensor<273x27x183xf32>
    %3 = tosa.reciprocal %2 : (tensor<273x27x183xf32>) -> tensor<273x27x183xf32>
    %4 = tosa.greater %3, %3 : (tensor<273x27x183xf32>, tensor<273x27x183xf32>) -> tensor<273x27x183xi1>
    %5 = tosa.identity %4 : (tensor<273x27x183xi1>) -> tensor<273x27x183xi1>
    return %5 : tensor<273x27x183xi1>
  }
}
