module {
  func.func @main(%arg0: tensor<68x9x71x39xi16>, %arg1: tensor<29x78x27x30x68xf32>, %arg2: tensor<29x1x27x30x68xf32>, %arg3: tensor<37x39x59x85x61xi32>, %arg4: tensor<1x39x1x85x61xi32>) -> (tensor<136x9x142x39xi16>, tensor<29x78x27x30x68xi1>, tensor<37x39x59x85x61xi32>) {
    %t_0 = tosa.const_shape {values = dense<[ 2, 1, 2, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.tile %arg0, %t_0 : (tensor<68x9x71x39xi16>, !tosa.shape<4>) -> tensor<136x9x142x39xi16>
    %1 = tosa.greater %arg1, %arg2 : (tensor<29x78x27x30x68xf32>, tensor<29x1x27x30x68xf32>) -> tensor<29x78x27x30x68xi1>
    %2 = tosa.minimum %arg3, %arg4 : (tensor<37x39x59x85x61xi32>, tensor<1x39x1x85x61xi32>) -> tensor<37x39x59x85x61xi32>
    return %0, %1, %2 : tensor<136x9x142x39xi16>, tensor<29x78x27x30x68xi1>, tensor<37x39x59x85x61xi32>
  }
}
