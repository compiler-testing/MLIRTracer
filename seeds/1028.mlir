module {
  func.func @main(%arg0: tensor<13x73xf32>) -> tensor<13x1xf32> {
    %0 = tosa.log %arg0 : (tensor<13x73xf32>) -> tensor<13x73xf32>
    %1 = tosa.reduce_product %0 {axis = 1 : i32} : (tensor<13x73xf32>) -> tensor<13x1xf32>
    return %1 : tensor<13x1xf32>
  }
}
