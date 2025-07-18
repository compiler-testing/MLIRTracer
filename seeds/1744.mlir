module {
  func.func @main(%arg0: tensor<56x38x28xf32>, %arg1: tensor<i8>, %arg2: tensor<i8>) -> (tensor<56x38x28xf32>, tensor<i8>) {
    %0 = tosa.floor %arg0 : (tensor<56x38x28xf32>) -> tensor<56x38x28xf32>
    %1 = tosa.rsqrt %0 : (tensor<56x38x28xf32>) -> tensor<56x38x28xf32>
    %2 = tosa.bitwise_xor %arg1, %arg2 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    return %1, %2 : tensor<56x38x28xf32>, tensor<i8>
  }
}
