module {
  func.func @main(%arg0: tensor<78xi64>, %arg1: tensor<93x94xf32>) -> (tensor<i32>, tensor<93x94xf32>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<78xi64>) -> tensor<i32>
    %1 = tosa.tanh %arg1 : (tensor<93x94xf32>) -> tensor<93x94xf32>
    %2 = tosa.bitwise_xor %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = tosa.bitwise_xor %2, %2 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = tosa.log %1 : (tensor<93x94xf32>) -> tensor<93x94xf32>
    return %3, %4 : tensor<i32>, tensor<93x94xf32>
  }
}
