module {
  func.func @main(%arg0: tensor<35x20x59x63xi32>, %arg1: tensor<4x91x88x18xi1>) -> (tensor<35x63x20x1xi32>, tensor<2x2x12x1xi1>) {
    %0 = "tosa.const"() {values = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 0, 3, 1, 2>} : (tensor<35x20x59x63xi32>) -> tensor<35x63x20x59xi32>
    %2 = tosa.intdiv %1, %1 : (tensor<35x63x20x59xi32>, tensor<35x63x20x59xi32>) -> tensor<35x63x20x59xi32>
    %3 = tosa.intdiv %2, %1 : (tensor<35x63x20x59xi32>, tensor<35x63x20x59xi32>) -> tensor<35x63x20x59xi32>
    %4 = tosa.reduce_all %arg1 {axis = 3 : i32} : (tensor<4x91x88x18xi1>) -> tensor<4x91x88x1xi1>
    %5 = tosa.identity %4 : (tensor<4x91x88x1xi1>) -> tensor<4x91x88x1xi1>
    %6 = tosa.logical_right_shift %3, %2 : (tensor<35x63x20x59xi32>, tensor<35x63x20x59xi32>) -> tensor<35x63x20x59xi32>
    %7 = tosa.intdiv %6, %6 : (tensor<35x63x20x59xi32>, tensor<35x63x20x59xi32>) -> tensor<35x63x20x59xi32>
    %8 = tosa.reduce_min %7 {axis = 3 : i32} : (tensor<35x63x20x59xi32>) -> tensor<35x63x20x1xi32>
    %s_9_start = tosa.const_shape {values = dense<[ 2, 2, 1, 0 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_9_size = tosa.const_shape {values = dense<[ 2, 2, 12, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %9 = tosa.slice %5, %s_9_start, %s_9_size : (tensor<4x91x88x1xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<2x2x12x3xi1>
    %10 = tosa.reduce_sum %9 {axis = 3 : i32} : (tensor<2x2x12x3xi1>) -> tensor<2x2x12x1xi1>
    return %8, %10 : tensor<35x63x20x1xi32>, tensor<2x2x12x1xi1>
  }
}
