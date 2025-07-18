module {
  func.func @main(%arg0: tensor<72x94x10xf32>) -> (tensor<72x94x10xi1>, tensor<216x282x10xi1>, tensor<72x94x10xi1>) {
    %0 = tosa.reciprocal %arg0 : (tensor<72x94x10xf32>) -> tensor<72x94x10xf32>
    %1 = tosa.ceil %0 : (tensor<72x94x10xf32>) -> tensor<72x94x10xf32>
    %2 = tosa.greater_equal %1, %1 : (tensor<72x94x10xf32>, tensor<72x94x10xf32>) -> tensor<72x94x10xi1>
    %3 = tosa.equal %0, %1 : (tensor<72x94x10xf32>, tensor<72x94x10xf32>) -> tensor<72x94x10xi1>
    %t_4 = tosa.const_shape {values = dense<[ 3, 3, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.tile %2, %t_4 : (tensor<72x94x10xi1>, !tosa.shape<3>) -> tensor<216x282x10xi1>
    %5 = tosa.equal %1, %1 : (tensor<72x94x10xf32>, tensor<72x94x10xf32>) -> tensor<72x94x10xi1>
    return %3, %4, %5 : tensor<72x94x10xi1>, tensor<216x282x10xi1>, tensor<72x94x10xi1>
  }
}
