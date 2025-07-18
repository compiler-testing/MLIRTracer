module {
  func.func @main(%arg0: tensor<68x45x1xf32>, %arg1: tensor<6x52x60x11x30xi1>) -> (tensor<12x52x60x11x30xi1>, tensor<68x1xi1>) {
    %0 = tosa.exp %arg0 : (tensor<68x45x1xf32>) -> tensor<68x45x1xf32>
    %1 = tosa.reciprocal %0 : (tensor<68x45x1xf32>) -> tensor<68x45x1xf32>
    %2 = tosa.reciprocal %1 : (tensor<68x45x1xf32>) -> tensor<68x45x1xf32>
    %3 = tosa.logical_not %arg1 : (tensor<6x52x60x11x30xi1>) -> tensor<6x52x60x11x30xi1>
    %4 = tosa.concat %3, %3 {axis = 0 : i32} : (tensor<6x52x60x11x30xi1>, tensor<6x52x60x11x30xi1>) -> tensor<12x52x60x11x30xi1>
    %5 = tosa.tanh %2 : (tensor<68x45x1xf32>) -> tensor<68x45x1xf32>
    %6 = tosa.reduce_sum %5 {axis = 1 : i32} : (tensor<68x45x1xf32>) -> tensor<68x1x1xf32>
    %7 = tosa.reverse %6 {axis = 2 : i32} : (tensor<68x1x1xf32>) -> tensor<68x1x1xf32>
    %8 = tosa.argmax %7 {axis = 1 : i32} : (tensor<68x1x1xf32>) -> tensor<68x1xi32>
    %9 = tosa.greater_equal %8, %8 : (tensor<68x1xi32>, tensor<68x1xi32>) -> tensor<68x1xi1>
    return %4, %9 : tensor<12x52x60x11x30xi1>, tensor<68x1xi1>
  }
}
