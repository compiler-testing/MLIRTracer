module {
  func.func @main(%arg0: tensor<14xf32>, %arg1: tensor<14xf32>, %arg2: tensor<66xi1>, %arg3: tensor<1xi1>) -> (tensor<42xf32>, tensor<66xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<14xf32>, tensor<14xf32>) -> tensor<14xf32>
    %t_1 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %1 = tosa.tile %0, %t_1 : (tensor<14xf32>, !tosa.shape<1>) -> tensor<42xf32>
    %2 = tosa.exp %1 : (tensor<42xf32>) -> tensor<42xf32>
    %3 = tosa.sigmoid %2 : (tensor<42xf32>) -> tensor<42xf32>
    %4 = tosa.logical_or %arg2, %arg3 : (tensor<66xi1>, tensor<1xi1>) -> tensor<66xi1>
    %5 = tosa.bitwise_not %4 : (tensor<66xi1>) -> tensor<66xi1>
    return %3, %5 : tensor<42xf32>, tensor<66xi1>
  }
}
