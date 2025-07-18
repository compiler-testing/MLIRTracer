module {
  func.func @main(%arg0: tensor<52x27xi32>) -> tensor<1x27xi32> {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<52x27xi32>) -> tensor<1x27xi32>
    return %0 : tensor<1x27xi32>
  }
}
