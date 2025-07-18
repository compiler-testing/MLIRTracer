module {
  func.func @main(%arg0: tensor<44x81x45xf32>) -> tensor<44x45xi32> {
    %0 = tosa.tanh %arg0 : (tensor<44x81x45xf32>) -> tensor<44x81x45xf32>
    %1 = tosa.sub %0, %0 : (tensor<44x81x45xf32>, tensor<44x81x45xf32>) -> tensor<44x81x45xf32>
    %2 = tosa.exp %1 : (tensor<44x81x45xf32>) -> tensor<44x81x45xf32>
    %3 = tosa.clamp %2 {min_val = 3.200000e+01 : f32, max_val = 1.260000e+02 : f32} : (tensor<44x81x45xf32>) -> tensor<44x81x45xf32>
    %4 = tosa.pow %3, %2 : (tensor<44x81x45xf32>, tensor<44x81x45xf32>) -> tensor<44x81x45xf32>
    %5 = tosa.argmax %4 {axis = 1 : i32} : (tensor<44x81x45xf32>) -> tensor<44x45xi32>
    %6 = tosa.bitwise_or %5, %5 : (tensor<44x45xi32>, tensor<44x45xi32>) -> tensor<44x45xi32>
    %7 = tosa.clz %6 : (tensor<44x45xi32>) -> tensor<44x45xi32>
    return %7 : tensor<44x45xi32>
  }
}
