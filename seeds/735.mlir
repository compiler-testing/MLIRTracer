module {
  func.func @main(%arg0: tensor<49xf32>) -> tensor<49xf32> {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<49xf32>) -> tensor<49xf32>
    %1 = tosa.exp %0 : (tensor<49xf32>) -> tensor<49xf32>
    return %1 : tensor<49xf32>
  }
}
