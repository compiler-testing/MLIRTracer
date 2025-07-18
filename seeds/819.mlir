module {
  func.func @main(%arg0: tensor<14x81x68x64x85xf32>, %arg1: tensor<1x81x68x1x1xf32>, %arg2: tensor<32xf32>) -> (tensor<14x81x68x64x85xf32>, tensor<1xf32>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<14x81x68x64x85xf32>, tensor<1x81x68x1x1xf32>) -> tensor<14x81x68x64x85xf32>
    %1 = tosa.reduce_product %arg2 {axis = 0 : i32} : (tensor<32xf32>) -> tensor<1xf32>
    return %0, %1 : tensor<14x81x68x64x85xf32>, tensor<1xf32>
  }
}
