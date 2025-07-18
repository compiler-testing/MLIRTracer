module {
  func.func @main(%arg0: tensor<31x46x42x16x89xi8>) -> tensor<31x46x42x16x89xi8> {
    %0 = tosa.bitwise_not %arg0 : (tensor<31x46x42x16x89xi8>) -> tensor<31x46x42x16x89xi8>
    return %0 : tensor<31x46x42x16x89xi8>
  }
}
