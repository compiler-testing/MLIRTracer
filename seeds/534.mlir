module {
  func.func @main(%arg0: tensor<85x47xi32>, %arg1: tensor<1x47xi32>, %arg2: tensor<75x13xf32>, %arg3: tensor<75x13xf32>) -> (tensor<85x47xi32>, tensor<75x13xf32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<85x47xi32>, tensor<1x47xi32>) -> tensor<85x47xi32>
    %1 = tosa.pow %arg2, %arg3 : (tensor<75x13xf32>, tensor<75x13xf32>) -> tensor<75x13xf32>
    return %0, %1 : tensor<85x47xi32>, tensor<75x13xf32>
  }
}
