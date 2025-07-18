module {
  func.func @main(%arg0: tensor<22x85xi64>, %arg1: tensor<57x73x75x58xf32>) -> (tensor<57x73x75x58xf32>, tensor<1x85xi64>) {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<22x85xi64>) -> tensor<1x85xi64>
    %1 = tosa.add %0, %0 : (tensor<1x85xi64>, tensor<1x85xi64>) -> tensor<1x85xi64>
    %2 = tosa.floor %arg1 : (tensor<57x73x75x58xf32>) -> tensor<57x73x75x58xf32>
    %3 = tosa.logical_left_shift %1, %1 : (tensor<1x85xi64>, tensor<1x85xi64>) -> tensor<1x85xi64>
    return %2, %3 : tensor<57x73x75x58xf32>, tensor<1x85xi64>
  }
}
