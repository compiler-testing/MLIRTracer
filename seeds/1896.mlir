module {
  func.func @main(%arg0: tensor<26x2x38x84xi8>, %arg1: tensor<26x2x38x1xi8>) -> tensor<26x2x38x84xi8> {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<26x2x38x84xi8>, tensor<26x2x38x1xi8>) -> tensor<26x2x38x84xi8>
    %1 = tosa.bitwise_and %0, %0 : (tensor<26x2x38x84xi8>, tensor<26x2x38x84xi8>) -> tensor<26x2x38x84xi8>
    return %1 : tensor<26x2x38x84xi8>
  }
}
