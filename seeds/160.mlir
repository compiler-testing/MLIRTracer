module {
  func.func @main(%arg0: tensor<78x5x85x53x28xi32>) -> tensor<6x8x1x3x7xi1> {
    %s_0_start = tosa.const_shape {values = dense<[ 71, 4, 14, 41, 22 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_0_size = tosa.const_shape {values = dense<[ 7, 1, 3, 8, 6 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<78x5x85x53x28xi32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<7x1x3x8x6xi32>
    %1 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 4, 3, 1, 2, 0>} : (tensor<7x1x3x8x6xi32>) -> tensor<6x8x1x3x7xi32>
    %3 = tosa.greater %2, %2 : (tensor<6x8x1x3x7xi32>, tensor<6x8x1x3x7xi32>) -> tensor<6x8x1x3x7xi1>
    return %3 : tensor<6x8x1x3x7xi1>
  }
}
