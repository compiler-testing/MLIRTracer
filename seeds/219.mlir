module {
  func.func @main(%arg0: tensor<34xi8>, %arg1: tensor<34xi8>, %arg2: tensor<17xi32>, %arg3: tensor<17xi32>) -> (tensor<17xi32>, tensor<34xi8>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<34xi8>, tensor<34xi8>) -> tensor<34xi8>
    %1 = tosa.clamp %0 {min_val = -28 : i8, max_val = 73 : i8} : (tensor<34xi8>) -> tensor<34xi8>
    %2 = tosa.intdiv %arg2, %arg3 : (tensor<17xi32>, tensor<17xi32>) -> tensor<17xi32>
    %3 = tosa.maximum %1, %0 : (tensor<34xi8>, tensor<34xi8>) -> tensor<34xi8>
    return %2, %3 : tensor<17xi32>, tensor<34xi8>
  }
}
