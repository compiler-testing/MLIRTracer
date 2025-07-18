module {
  func.func @main(%arg0: tensor<21x18x59x27x34x14xf32>, %arg1: tensor<48x25xi64>) -> (tensor<9x2x5x5x9x8xf32>, tensor<1x25xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<21x18x59x27x34x14xf32>) -> tensor<21x18x59x27x34x14xf32>
    %s_1_start = tosa.const_shape {values = dense<[ 3, 1, 10, 18, 8, 6 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_1_size = tosa.const_shape {values = dense<[ 9, 2, 5, 5, 9, 8 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<21x18x59x27x34x14xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<9x2x5x5x9x8xf32>
    %2 = tosa.reduce_max %arg1 {axis = 0 : i32} : (tensor<48x25xi64>) -> tensor<1x25xi64>
    %3 = tosa.greater_equal %2, %2 : (tensor<1x25xi64>, tensor<1x25xi64>) -> tensor<1x25xi1>
    return %1, %3 : tensor<9x2x5x5x9x8xf32>, tensor<1x25xi1>
  }
}
