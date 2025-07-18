module {
  func.func @main(%arg0: tensor<41x66xi32>) -> tensor<41x66xi32> {
    %0 = tosa.clz %arg0 : (tensor<41x66xi32>) -> tensor<41x66xi32>
    %1 = tosa.clz %0 : (tensor<41x66xi32>) -> tensor<41x66xi32>
    %2 = tosa.bitwise_not %1 : (tensor<41x66xi32>) -> tensor<41x66xi32>
    return %2 : tensor<41x66xi32>
  }
}
