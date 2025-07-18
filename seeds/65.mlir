module {
  func.func @main(%arg0: tensor<28x15x92x51xf32>) -> (tensor<28x15x92x51xf32>, tensor<28x15x92x51xf32>, tensor<28x15x1x51xi1>) {
    %0 = tosa.log %arg0 : (tensor<28x15x92x51xf32>) -> tensor<28x15x92x51xf32>
    %1 = tosa.equal %0, %0 : (tensor<28x15x92x51xf32>, tensor<28x15x92x51xf32>) -> tensor<28x15x92x51xi1>
    %2 = tosa.bitwise_xor %1, %1 : (tensor<28x15x92x51xi1>, tensor<28x15x92x51xi1>) -> tensor<28x15x92x51xi1>
    %3 = tosa.minimum %0, %0 : (tensor<28x15x92x51xf32>, tensor<28x15x92x51xf32>) -> tensor<28x15x92x51xf32>
    %4 = tosa.reduce_any %2 {axis = 2 : i32} : (tensor<28x15x92x51xi1>) -> tensor<28x15x1x51xi1>
    %5 = tosa.arithmetic_right_shift %4, %4 {round = false} : (tensor<28x15x1x51xi1>, tensor<28x15x1x51xi1>) -> tensor<28x15x1x51xi1>
    %6 = tosa.rsqrt %3 : (tensor<28x15x92x51xf32>) -> tensor<28x15x92x51xf32>
    %7 = tosa.sub %5, %5 : (tensor<28x15x1x51xi1>, tensor<28x15x1x51xi1>) -> tensor<28x15x1x51xi1>
    %8 = tosa.floor %3 : (tensor<28x15x92x51xf32>) -> tensor<28x15x92x51xf32>
    %9 = tosa.reverse %7 {axis = 0 : i32} : (tensor<28x15x1x51xi1>) -> tensor<28x15x1x51xi1>
    %10 = tosa.bitwise_and %9, %7 : (tensor<28x15x1x51xi1>, tensor<28x15x1x51xi1>) -> tensor<28x15x1x51xi1>
    return %6, %8, %10 : tensor<28x15x92x51xf32>, tensor<28x15x92x51xf32>, tensor<28x15x1x51xi1>
  }
}
