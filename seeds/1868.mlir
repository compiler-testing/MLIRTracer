module {
  func.func @main(%arg0: tensor<22xi8>, %arg1: tensor<1xi8>, %arg2: tensor<74x22x91x50xi1>) -> (tensor<22xi8>, tensor<74x22x91x1xi1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<22xi8>, tensor<1xi8>) -> tensor<22xi8>
    %1 = tosa.reduce_all %arg2 {axis = 3 : i32} : (tensor<74x22x91x50xi1>) -> tensor<74x22x91x1xi1>
    return %0, %1 : tensor<22xi8>, tensor<74x22x91x1xi1>
  }
}
