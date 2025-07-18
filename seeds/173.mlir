module {
  func.func @main(%arg0: tensor<99x48x35xi1>, %arg1: tensor<99x48x35xi1>, %arg2: tensor<60x9x42xf32>) -> (tensor<60x9x42xf32>, tensor<1x66x6x420xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<99x48x35xi1>, tensor<99x48x35xi1>) -> tensor<99x48x35xi1>
    %1 = tosa.rsqrt %arg2 : (tensor<60x9x42xf32>) -> tensor<60x9x42xf32>
    %r_2 = tosa.const_shape {values = dense<[ 1, 66, 6, 420 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.reshape %0, %r_2 : (tensor<99x48x35xi1>, !tosa.shape<4>) -> tensor<1x66x6x420xi1>
    return %1, %2 : tensor<60x9x42xf32>, tensor<1x66x6x420xi1>
  }
}
