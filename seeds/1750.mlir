module {
  func.func @main(%arg0: tensor<45x12x31x44x80x82xf32>, %arg1: tensor<96x15x55xi1>) -> (tensor<45x12x31x44x80x82xf32>, tensor<96x55xi32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<45x12x31x44x80x82xf32>) -> tensor<45x12x31x44x80x82xf32>
    %1 = tosa.identity %0 : (tensor<45x12x31x44x80x82xf32>) -> tensor<45x12x31x44x80x82xf32>
    %2 = tosa.argmax %arg1 {axis = 1 : i32} : (tensor<96x15x55xi1>) -> tensor<96x55xi32>
    return %1, %2 : tensor<45x12x31x44x80x82xf32>, tensor<96x55xi32>
  }
}
