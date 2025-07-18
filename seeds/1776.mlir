module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<59x76x94xf32>, %arg2: tensor<59x76x94xf32>, %arg3: tensor<34x15x46x16xi1>) -> (tensor<1x1xi32>, tensor<59x76x94xf32>, tensor<34x15x1x16xi1>) {
    %r_0 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<i32>, !tosa.shape<2>) -> tensor<1x1xi32>
    %1 = tosa.add %0, %0 : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
    %2 = tosa.pow %arg1, %arg2 : (tensor<59x76x94xf32>, tensor<59x76x94xf32>) -> tensor<59x76x94xf32>
    %3 = tosa.reduce_all %arg3 {axis = 2 : i32} : (tensor<34x15x46x16xi1>) -> tensor<34x15x1x16xi1>
    return %1, %2, %3 : tensor<1x1xi32>, tensor<59x76x94xf32>, tensor<34x15x1x16xi1>
  }
}
