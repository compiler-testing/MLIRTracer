module {
  func.func @main(%arg0: tensor<37x88x95x19x8x23xi1>, %arg1: tensor<29x72x55x59x2x6xf32>) -> (tensor<37x88x95x19x8x23xi1>, tensor<29x72x55x59x2x6xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<37x88x95x19x8x23xi1>) -> tensor<37x88x95x19x8x23xi1>
    %1 = tosa.tanh %arg1 : (tensor<29x72x55x59x2x6xf32>) -> tensor<29x72x55x59x2x6xf32>
    return %0, %1 : tensor<37x88x95x19x8x23xi1>, tensor<29x72x55x59x2x6xf32>
  }
}
