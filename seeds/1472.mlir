module {
  func.func @main(%arg0: tensor<8xf32>) -> tensor<1xf32> {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<8xf32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}
