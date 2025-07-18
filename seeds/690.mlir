module {
  func.func @main(%arg0: tensor<4xi16>, %arg1: tensor<4xi16>, %arg2: tensor<87x76x68x32x15xf32>) -> (tensor<4xi16>, tensor<87x76x68x32x15xf32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<4xi16>, tensor<4xi16>) -> tensor<4xi16>
    %1 = tosa.identity %0 : (tensor<4xi16>) -> tensor<4xi16>
    %2 = tosa.ceil %arg2 : (tensor<87x76x68x32x15xf32>) -> tensor<87x76x68x32x15xf32>
    return %1, %2 : tensor<4xi16>, tensor<87x76x68x32x15xf32>
  }
}
