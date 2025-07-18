module {
  func.func @main(%arg0: tensor<5x92x36x31xf32>, %arg1: tensor<92x30x22x9xi64>, %arg2: tensor<92x30x22x9xi64>) -> (tensor<5x1x36x31xf32>, tensor<92x30x1x9xi64>) {
    %0 = tosa.reduce_min %arg0 {axis = 1 : i32} : (tensor<5x92x36x31xf32>) -> tensor<5x1x36x31xf32>
    %1 = tosa.arithmetic_right_shift %arg1, %arg2 {round = false} : (tensor<92x30x22x9xi64>, tensor<92x30x22x9xi64>) -> tensor<92x30x22x9xi64>
    %2 = tosa.reduce_sum %1 {axis = 2 : i32} : (tensor<92x30x22x9xi64>) -> tensor<92x30x1x9xi64>
    return %0, %2 : tensor<5x1x36x31xf32>, tensor<92x30x1x9xi64>
  }
}
