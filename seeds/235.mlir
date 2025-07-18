module {
  func.func @main(%arg0: tensor<79xi16>, %arg1: tensor<79xi16>, %arg2: tensor<88x4xi1>, %arg3: tensor<88x1xi1>) -> (tensor<i32>, tensor<4xi32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<79xi16>, tensor<79xi16>) -> tensor<79xi16>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<79xi16>, tensor<79xi16>) -> tensor<79xi16>
    %2 = tosa.identity %1 : (tensor<79xi16>) -> tensor<79xi16>
    %3 = tosa.argmax %2 {axis = 0 : i32} : (tensor<79xi16>) -> tensor<i32>
    %4 = tosa.logical_xor %arg2, %arg3 : (tensor<88x4xi1>, tensor<88x1xi1>) -> tensor<88x4xi1>
    %5 = tosa.argmax %4 {axis = 0 : i32} : (tensor<88x4xi1>) -> tensor<4xi32>
    %6 = tosa.logical_left_shift %5, %5 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    return %3, %6 : tensor<i32>, tensor<4xi32>
  }
}
