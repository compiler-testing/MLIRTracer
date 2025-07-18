module {
  func.func @main(%arg0: tensor<6x10x12x23xi32>, %arg1: tensor<6x1x1x23xi32>, %arg2: tensor<59x72x59x10x27x27xf32>) -> (tensor<59x72x59x10x27x27xi1>, tensor<59x72x59x10x27x27xf32>, tensor<6x10x1x23xi32>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<6x10x12x23xi32>, tensor<6x1x1x23xi32>) -> tensor<6x10x12x23xi32>
    %1 = tosa.log %arg2 : (tensor<59x72x59x10x27x27xf32>) -> tensor<59x72x59x10x27x27xf32>
    %2 = tosa.bitwise_not %0 : (tensor<6x10x12x23xi32>) -> tensor<6x10x12x23xi32>
    %3 = tosa.bitwise_and %2, %2 : (tensor<6x10x12x23xi32>, tensor<6x10x12x23xi32>) -> tensor<6x10x12x23xi32>
    %4 = tosa.maximum %3, %0 : (tensor<6x10x12x23xi32>, tensor<6x10x12x23xi32>) -> tensor<6x10x12x23xi32>
    %5 = tosa.arithmetic_right_shift %4, %0 {round = false} : (tensor<6x10x12x23xi32>, tensor<6x10x12x23xi32>) -> tensor<6x10x12x23xi32>
    %6 = tosa.greater %1, %1 : (tensor<59x72x59x10x27x27xf32>, tensor<59x72x59x10x27x27xf32>) -> tensor<59x72x59x10x27x27xi1>
    %7 = tosa.log %1 : (tensor<59x72x59x10x27x27xf32>) -> tensor<59x72x59x10x27x27xf32>
    %8 = tosa.minimum %5, %3 : (tensor<6x10x12x23xi32>, tensor<6x10x12x23xi32>) -> tensor<6x10x12x23xi32>
    %9 = tosa.log %7 : (tensor<59x72x59x10x27x27xf32>) -> tensor<59x72x59x10x27x27xf32>
    %10 = tosa.ceil %9 : (tensor<59x72x59x10x27x27xf32>) -> tensor<59x72x59x10x27x27xf32>
    %11 = tosa.bitwise_and %6, %6 : (tensor<59x72x59x10x27x27xi1>, tensor<59x72x59x10x27x27xi1>) -> tensor<59x72x59x10x27x27xi1>
    %12 = tosa.reciprocal %10 : (tensor<59x72x59x10x27x27xf32>) -> tensor<59x72x59x10x27x27xf32>
    %13 = tosa.reduce_sum %8 {axis = 2 : i32} : (tensor<6x10x12x23xi32>) -> tensor<6x10x1x23xi32>
    return %11, %12, %13 : tensor<59x72x59x10x27x27xi1>, tensor<59x72x59x10x27x27xf32>, tensor<6x10x1x23xi32>
  }
}
