module {
  func.func @main(%arg0: tensor<40x16xi1>, %arg1: tensor<76x94x20x74x34xf32>) -> (tensor<40x1xi1>, tensor<76x94x20x74x34xf32>) {
    %0 = tosa.reduce_any %arg0 {axis = 1 : i32} : (tensor<40x16xi1>) -> tensor<40x1xi1>
    %1 = tosa.reciprocal %arg1 : (tensor<76x94x20x74x34xf32>) -> tensor<76x94x20x74x34xf32>
    %2 = tosa.pow %1, %1 : (tensor<76x94x20x74x34xf32>, tensor<76x94x20x74x34xf32>) -> tensor<76x94x20x74x34xf32>
    return %0, %2 : tensor<40x1xi1>, tensor<76x94x20x74x34xf32>
  }
}
