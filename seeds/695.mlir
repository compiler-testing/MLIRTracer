module {
  func.func @main(%arg0: tensor<57x4x46x25xi1>, %arg1: tensor<57x1x1x1xi1>, %arg2: tensor<34x14x16xf32>) -> (tensor<5x12x11xf32>, tensor<57x25x4x46xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<57x4x46x25xi1>, tensor<57x1x1x1xi1>) -> tensor<57x4x46x25xi1>
    %1 = tosa.rsqrt %arg2 : (tensor<34x14x16xf32>) -> tensor<34x14x16xf32>
    %2 = tosa.logical_or %0, %0 : (tensor<57x4x46x25xi1>, tensor<57x4x46x25xi1>) -> tensor<57x4x46x25xi1>
    %s_3_start = tosa.const_shape {values = dense<[ 19, 2, 5 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_3_size = tosa.const_shape {values = dense<[ 5, 12, 11 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %3 = tosa.slice %1, %s_3_start, %s_3_size : (tensor<34x14x16xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<5x12x11xf32>
    %4 = "tosa.const"() {values = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %5 = tosa.transpose %2 {perms = array<i32: 0, 3, 1, 2>} : (tensor<57x4x46x25xi1>) -> tensor<57x25x4x46xi1>
    return %3, %5 : tensor<5x12x11xf32>, tensor<57x25x4x46xi1>
  }
}
