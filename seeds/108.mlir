module {
  func.func @main(%arg0: tensor<54x8xf32>) -> (tensor<54x8xf32>, tensor<1x8xi1>) {
    %0 = tosa.reciprocal %arg0 : (tensor<54x8xf32>) -> tensor<54x8xf32>
    %1 = tosa.log %0 : (tensor<54x8xf32>) -> tensor<54x8xf32>
    %2 = tosa.pow %1, %1 : (tensor<54x8xf32>, tensor<54x8xf32>) -> tensor<54x8xf32>
    %3 = tosa.greater_equal %2, %1 : (tensor<54x8xf32>, tensor<54x8xf32>) -> tensor<54x8xi1>
    %4 = tosa.pow %0, %1 : (tensor<54x8xf32>, tensor<54x8xf32>) -> tensor<54x8xf32>
    %5 = tosa.reduce_any %3 {axis = 0 : i32} : (tensor<54x8xi1>) -> tensor<1x8xi1>
    %6 = tosa.add %5, %5 : (tensor<1x8xi1>, tensor<1x8xi1>) -> tensor<1x8xi1>
    return %4, %6 : tensor<54x8xf32>, tensor<1x8xi1>
  }
}
