module {
  func.func @main(%arg0: tensor<1x33x46x74xf32>, %arg1: tensor<85x59x61x71x92x56xi1>, %arg2: tensor<1x59x1x1x92x56xi1>) -> (tensor<85x59x61x71x92x56xi1>, tensor<112332xf32>) {
    %0 = tosa.log %arg0 : (tensor<1x33x46x74xf32>) -> tensor<1x33x46x74xf32>
    %1 = tosa.logical_or %arg1, %arg2 : (tensor<85x59x61x71x92x56xi1>, tensor<1x59x1x1x92x56xi1>) -> tensor<85x59x61x71x92x56xi1>
    %r_2 = tosa.const_shape {values = dense<[ 112332 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.reshape %0, %r_2 : (tensor<1x33x46x74xf32>, !tosa.shape<1>) -> tensor<112332xf32>
    return %1, %2 : tensor<85x59x61x71x92x56xi1>, tensor<112332xf32>
  }
}
