module {
  func.func @main(%arg0: tensor<100x77x53x31xf32>) -> tensor<1x77x53x31xf32> {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<100x77x53x31xf32>) -> tensor<1x77x53x31xf32>
    return %0 : tensor<1x77x53x31xf32>
  }
}
