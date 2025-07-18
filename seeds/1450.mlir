module {
  func.func @main(%arg0: tensor<61x45xi8>) -> tensor<61xi32> {
    %0 = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<61x45xi8>) -> tensor<61xi32>
    %1 = tosa.bitwise_or %0, %0 : (tensor<61xi32>, tensor<61xi32>) -> tensor<61xi32>
    %2 = tosa.intdiv %1, %0 : (tensor<61xi32>, tensor<61xi32>) -> tensor<61xi32>
    return %2 : tensor<61xi32>
  }
}
