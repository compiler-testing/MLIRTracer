module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>, %arg2: tensor<76x29xf32>) -> (tensor<i8>, tensor<76x29xf32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %1 = tosa.clamp %0 {min_val = 16 : i8, max_val = 31 : i8} : (tensor<i8>) -> tensor<i8>
    %2 = tosa.ceil %arg2 : (tensor<76x29xf32>) -> tensor<76x29xf32>
    return %1, %2 : tensor<i8>, tensor<76x29xf32>
  }
}
