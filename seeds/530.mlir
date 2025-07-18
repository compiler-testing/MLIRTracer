module {
  func.func @main(%arg0: tensor<76x27x28xi32>) -> tensor<76x1x1xi1> {
    %0 = tosa.reduce_min %arg0 {axis = 2 : i32} : (tensor<76x27x28xi32>) -> tensor<76x27x1xi32>
    %1 = tosa.clamp %0 {min_val = 17 : i32, max_val = 115 : i32} : (tensor<76x27x1xi32>) -> tensor<76x27x1xi32>
    %2 = tosa.greater_equal %1, %0 : (tensor<76x27x1xi32>, tensor<76x27x1xi32>) -> tensor<76x27x1xi1>
    %3 = tosa.bitwise_not %2 : (tensor<76x27x1xi1>) -> tensor<76x27x1xi1>
    %4 = tosa.logical_right_shift %3, %2 : (tensor<76x27x1xi1>, tensor<76x27x1xi1>) -> tensor<76x27x1xi1>
    %5 = tosa.reduce_any %4 {axis = 1 : i32} : (tensor<76x27x1xi1>) -> tensor<76x1x1xi1>
    return %5 : tensor<76x1x1xi1>
  }
}
