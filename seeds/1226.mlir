module {
  func.func @main(%arg0: tensor<27x68x88x91xi8>, %arg1: tensor<56xf32>) -> (tensor<1x68x91xi32>, tensor<56xf32>) {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<27x68x88x91xi8>) -> tensor<1x68x88x91xi8>
    %1 = tosa.argmax %0 {axis = 2 : i32} : (tensor<1x68x88x91xi8>) -> tensor<1x68x91xi32>
    %2 = tosa.bitwise_or %1, %1 : (tensor<1x68x91xi32>, tensor<1x68x91xi32>) -> tensor<1x68x91xi32>
    %3 = tosa.abs %2 : (tensor<1x68x91xi32>) -> tensor<1x68x91xi32>
    %4 = tosa.reciprocal %arg1 : (tensor<56xf32>) -> tensor<56xf32>
    return %3, %4 : tensor<1x68x91xi32>, tensor<56xf32>
  }
}
