module {
  func.func @main(%arg0: tensor<23xf32>) -> (tensor<23xf32>, tensor<1xi1>) {
    %t_0 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.tile %arg0, %t_0 : (tensor<23xf32>, !tosa.shape<1>) -> tensor<23xf32>
    %1 = tosa.greater %0, %0 : (tensor<23xf32>, tensor<23xf32>) -> tensor<23xi1>
    %2 = tosa.logical_or %1, %1 : (tensor<23xi1>, tensor<23xi1>) -> tensor<23xi1>
    %3 = tosa.maximum %0, %0 : (tensor<23xf32>, tensor<23xf32>) -> tensor<23xf32>
    %4 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<23xi1>) -> tensor<1xi1>
    return %3, %4 : tensor<23xf32>, tensor<1xi1>
  }
}
