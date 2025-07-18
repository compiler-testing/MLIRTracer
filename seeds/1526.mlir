module {
  func.func @main(%arg0: tensor<49x53x7x13x94xf32>, %arg1: tensor<60x92xi1>, %arg2: tensor<60x92xi1>) -> (tensor<49x53x7x13x94xf32>, tensor<60x1xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<49x53x7x13x94xf32>) -> tensor<49x53x7x13x94xf32>
    %1 = tosa.identity %0 : (tensor<49x53x7x13x94xf32>) -> tensor<49x53x7x13x94xf32>
    %2 = tosa.pow %1, %1 : (tensor<49x53x7x13x94xf32>, tensor<49x53x7x13x94xf32>) -> tensor<49x53x7x13x94xf32>
    %3 = tosa.logical_or %arg1, %arg2 : (tensor<60x92xi1>, tensor<60x92xi1>) -> tensor<60x92xi1>
    %4 = tosa.reduce_any %3 {axis = 1 : i32} : (tensor<60x92xi1>) -> tensor<60x1xi1>
    return %2, %4 : tensor<49x53x7x13x94xf32>, tensor<60x1xi1>
  }
}
