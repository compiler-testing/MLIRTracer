module {
  func.func @main(%arg0: tensor<58xf32>) -> tensor<1xf32> {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<58xf32>) -> tensor<1xf32>
    %1 = tosa.identity %0 : (tensor<1xf32>) -> tensor<1xf32>
    return %1 : tensor<1xf32>
  }
}
