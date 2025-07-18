module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<74x73x6xi8>, %arg2: tensor<1x73x6xi8>) -> (tensor<i8>, tensor<74x73x6xi8>) {
    %0 = tosa.clamp %arg0 {min_val = 1 : i8, max_val = 83 : i8} : (tensor<i8>) -> tensor<i8>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %2 = tosa.maximum %arg1, %arg2 : (tensor<74x73x6xi8>, tensor<1x73x6xi8>) -> tensor<74x73x6xi8>
    return %1, %2 : tensor<i8>, tensor<74x73x6xi8>
  }
}
