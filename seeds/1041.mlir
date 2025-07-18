module {
  func.func @main(%arg0: tensor<44x46x97xi64>, %arg1: tensor<63x94xf32>, %arg2: tensor<44x30x86xi1>, %arg3: tensor<1x30x86xi1>) -> (tensor<196328xi64>, tensor<63x94xf32>, tensor<44x30x86xi1>) {
    %r_0 = tosa.const_shape {values = dense<[ 196328 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<44x46x97xi64>, !tosa.shape<1>) -> tensor<196328xi64>
    %1 = tosa.rsqrt %arg1 : (tensor<63x94xf32>) -> tensor<63x94xf32>
    %2 = tosa.logical_or %arg2, %arg3 : (tensor<44x30x86xi1>, tensor<1x30x86xi1>) -> tensor<44x30x86xi1>
    return %0, %1, %2 : tensor<196328xi64>, tensor<63x94xf32>, tensor<44x30x86xi1>
  }
}
