module {
  func.func @main(%arg0: tensor<52x44xf32>, %arg1: tensor<83x33x32xi1>) -> (tensor<52x1xf32>, tensor<83x33x1xi1>) {
    %0 = tosa.floor %arg0 : (tensor<52x44xf32>) -> tensor<52x44xf32>
    %1 = tosa.reduce_min %0 {axis = 1 : i32} : (tensor<52x44xf32>) -> tensor<52x1xf32>
    %2 = tosa.reduce_any %arg1 {axis = 2 : i32} : (tensor<83x33x32xi1>) -> tensor<83x33x1xi1>
    return %1, %2 : tensor<52x1xf32>, tensor<83x33x1xi1>
  }
}
