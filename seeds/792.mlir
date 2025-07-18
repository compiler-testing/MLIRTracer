module {
  func.func @main(%arg0: tensor<88x8x49xi64>, %arg1: tensor<83xi32>, %arg2: tensor<1xi32>, %arg3: tensor<f32>) -> (tensor<1x8x49xi1>, tensor<83xi32>, tensor<f32>) {
    %0 = tosa.identity %arg0 : (tensor<88x8x49xi64>) -> tensor<88x8x49xi64>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<88x8x49xi64>) -> tensor<1x8x49xi64>
    %2 = tosa.greater %1, %1 : (tensor<1x8x49xi64>, tensor<1x8x49xi64>) -> tensor<1x8x49xi1>
    %3 = tosa.intdiv %arg1, %arg2 : (tensor<83xi32>, tensor<1xi32>) -> tensor<83xi32>
    %4 = tosa.sigmoid %arg3 : (tensor<f32>) -> tensor<f32>
    return %2, %3, %4 : tensor<1x8x49xi1>, tensor<83xi32>, tensor<f32>
  }
}
