module {
  func.func @main(%arg0: tensor<56x15x88xi1>, %arg1: tensor<40x9x45xf32>, %arg2: tensor<40x1x45xf32>) -> (tensor<56x1x88xi1>, tensor<56x1x88xi1>, tensor<40x9x45xi1>) {
    %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<56x15x88xi1>) -> tensor<56x1x88xi1>
    %1 = tosa.minimum %arg1, %arg2 : (tensor<40x9x45xf32>, tensor<40x1x45xf32>) -> tensor<40x9x45xf32>
    %2 = tosa.minimum %1, %1 : (tensor<40x9x45xf32>, tensor<40x9x45xf32>) -> tensor<40x9x45xf32>
    %3 = tosa.identity %1 : (tensor<40x9x45xf32>) -> tensor<40x9x45xf32>
    %4 = tosa.exp %3 : (tensor<40x9x45xf32>) -> tensor<40x9x45xf32>
    %5 = tosa.reduce_any %0 {axis = 1 : i32} : (tensor<56x1x88xi1>) -> tensor<56x1x88xi1>
    %6 = tosa.logical_not %0 : (tensor<56x1x88xi1>) -> tensor<56x1x88xi1>
    %7 = tosa.greater_equal %4, %2 : (tensor<40x9x45xf32>, tensor<40x9x45xf32>) -> tensor<40x9x45xi1>
    return %5, %6, %7 : tensor<56x1x88xi1>, tensor<56x1x88xi1>, tensor<40x9x45xi1>
  }
}
