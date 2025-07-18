module {
  func.func @main(%arg0: tensor<55x69x36xf32>, %arg1: tensor<58x29xi1>) -> (tensor<55x69x36xf32>, tensor<58x1xi1>) {
    %0 = tosa.rsqrt %arg0 : (tensor<55x69x36xf32>) -> tensor<55x69x36xf32>
    %1 = tosa.floor %0 : (tensor<55x69x36xf32>) -> tensor<55x69x36xf32>
    %2 = tosa.reduce_any %arg1 {axis = 1 : i32} : (tensor<58x29xi1>) -> tensor<58x1xi1>
    return %1, %2 : tensor<55x69x36xf32>, tensor<58x1xi1>
  }
}
