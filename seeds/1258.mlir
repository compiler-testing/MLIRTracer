module {
  func.func @main(%arg0: tensor<72x9x79x97xi1>, %arg1: tensor<81x97x88xf32>) -> (tensor<81x97x176xf32>, tensor<1x9x79x1xi1>) {
    %0 = tosa.reduce_any %arg0 {axis = 3 : i32} : (tensor<72x9x79x97xi1>) -> tensor<72x9x79x1xi1>
    %1 = tosa.bitwise_not %0 : (tensor<72x9x79x1xi1>) -> tensor<72x9x79x1xi1>
    %2 = tosa.reciprocal %arg1 : (tensor<81x97x88xf32>) -> tensor<81x97x88xf32>
    %3 = tosa.concat %2, %2 {axis = 2 : i32} : (tensor<81x97x88xf32>, tensor<81x97x88xf32>) -> tensor<81x97x176xf32>
    %4 = tosa.reduce_any %1 {axis = 0 : i32} : (tensor<72x9x79x1xi1>) -> tensor<1x9x79x1xi1>
    %5 = tosa.bitwise_or %4, %4 : (tensor<1x9x79x1xi1>, tensor<1x9x79x1xi1>) -> tensor<1x9x79x1xi1>
    %6 = tosa.abs %5 : (tensor<1x9x79x1xi1>) -> tensor<1x9x79x1xi1>
    return %3, %6 : tensor<81x97x176xf32>, tensor<1x9x79x1xi1>
  }
}
