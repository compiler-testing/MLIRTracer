module {
  func.func @main(%arg0: tensor<96x4xi32>) -> tensor<1x1xi1> {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<96x4xi32>) -> tensor<1x4xi32>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<1x4xi32>) -> tensor<1x4xi32>
    %2 = tosa.reduce_sum %1 {axis = 1 : i32} : (tensor<1x4xi32>) -> tensor<1x1xi32>
    %3 = tosa.greater %2, %2 : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    return %3 : tensor<1x1xi1>
  }
}
