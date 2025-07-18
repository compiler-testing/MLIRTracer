module {
  func.func @main(%arg0: tensor<7x18x9x77x59xf32>, %arg1: tensor<42x71x58xi32>, %arg2: tensor<1x1x58xi32>) -> (tensor<42x71xi32>, tensor<7x18x9x77x59xf32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<7x18x9x77x59xf32>) -> tensor<7x18x9x77x59xf32>
    %1 = tosa.sigmoid %0 : (tensor<7x18x9x77x59xf32>) -> tensor<7x18x9x77x59xf32>
    %2 = tosa.bitwise_or %arg1, %arg2 : (tensor<42x71x58xi32>, tensor<1x1x58xi32>) -> tensor<42x71x58xi32>
    %3 = tosa.argmax %2 {axis = 2 : i32} : (tensor<42x71x58xi32>) -> tensor<42x71xi32>
    %4 = tosa.rsqrt %1 : (tensor<7x18x9x77x59xf32>) -> tensor<7x18x9x77x59xf32>
    return %3, %4 : tensor<42x71xi32>, tensor<7x18x9x77x59xf32>
  }
}
