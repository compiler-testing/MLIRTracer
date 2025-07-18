module {
  func.func @main(%arg0: tensor<20x50x9xi64>, %arg1: tensor<1x50x1xi64>, %arg2: tensor<9x20x45x19x41xf32>) -> (tensor<20x50x9xi1>, tensor<1x50x9xi1>, tensor<9x20x45x19x41xf32>, tensor<20x1x9xi1>, tensor<20x50x9xi1>, tensor<9x20x45x19x41xf32>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<20x50x9xi64>, tensor<1x50x1xi64>) -> tensor<20x50x9xi1>
    %1 = tosa.add %0, %0 : (tensor<20x50x9xi1>, tensor<20x50x9xi1>) -> tensor<20x50x9xi1>
    %2 = tosa.reduce_all %1 {axis = 1 : i32} : (tensor<20x50x9xi1>) -> tensor<20x1x9xi1>
    %3 = tosa.reciprocal %arg2 : (tensor<9x20x45x19x41xf32>) -> tensor<9x20x45x19x41xf32>
    %4 = tosa.exp %3 : (tensor<9x20x45x19x41xf32>) -> tensor<9x20x45x19x41xf32>
    %5 = tosa.pow %4, %4 : (tensor<9x20x45x19x41xf32>, tensor<9x20x45x19x41xf32>) -> tensor<9x20x45x19x41xf32>
    %6 = tosa.reverse %1 {axis = 2 : i32} : (tensor<20x50x9xi1>) -> tensor<20x50x9xi1>
    %7 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<20x50x9xi1>) -> tensor<1x50x9xi1>
    %8 = tosa.tanh %3 : (tensor<9x20x45x19x41xf32>) -> tensor<9x20x45x19x41xf32>
    %9 = tosa.logical_right_shift %2, %2 : (tensor<20x1x9xi1>, tensor<20x1x9xi1>) -> tensor<20x1x9xi1>
    %10 = tosa.arithmetic_right_shift %0, %1 {round = true} : (tensor<20x50x9xi1>, tensor<20x50x9xi1>) -> tensor<20x50x9xi1>
    %11 = tosa.rsqrt %5 : (tensor<9x20x45x19x41xf32>) -> tensor<9x20x45x19x41xf32>
    return %6, %7, %8, %9, %10, %11 : tensor<20x50x9xi1>, tensor<1x50x9xi1>, tensor<9x20x45x19x41xf32>, tensor<20x1x9xi1>, tensor<20x50x9xi1>, tensor<9x20x45x19x41xf32>
  }
}
