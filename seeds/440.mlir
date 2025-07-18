module {
  func.func @main(%arg0: tensor<56x92x12x81xi64>, %arg1: tensor<36x4xi1>) -> (tensor<56x1x1x81xi64>, tensor<1x4xi1>) {
    %0 = tosa.reduce_min %arg0 {axis = 1 : i32} : (tensor<56x92x12x81xi64>) -> tensor<56x1x12x81xi64>
    %1 = tosa.reduce_sum %0 {axis = 2 : i32} : (tensor<56x1x12x81xi64>) -> tensor<56x1x1x81xi64>
    %2 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<36x4xi1>) -> tensor<1x4xi1>
    return %1, %2 : tensor<56x1x1x81xi64>, tensor<1x4xi1>
  }
}
