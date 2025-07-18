module {
  func.func @main(%arg0: tensor<47x98x81x25x16xf32>) -> tensor<47x98x81x25x16xf32> {
    %0 = tosa.rsqrt %arg0 : (tensor<47x98x81x25x16xf32>) -> tensor<47x98x81x25x16xf32>
    %1 = tosa.floor %0 : (tensor<47x98x81x25x16xf32>) -> tensor<47x98x81x25x16xf32>
    return %1 : tensor<47x98x81x25x16xf32>
  }
}
