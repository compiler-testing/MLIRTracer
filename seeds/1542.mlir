module {
  func.func @main(%arg0: tensor<58xf32>, %arg1: tensor<1xf32>, %arg2: tensor<28x60xi1>, %arg3: tensor<1x60xi1>) -> (tensor<58xf32>, tensor<1x60xi1>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<58xf32>, tensor<1xf32>) -> tensor<58xf32>
    %1 = tosa.logical_or %arg2, %arg3 : (tensor<28x60xi1>, tensor<1x60xi1>) -> tensor<28x60xi1>
    %2 = tosa.logical_xor %1, %1 : (tensor<28x60xi1>, tensor<28x60xi1>) -> tensor<28x60xi1>
    %3 = tosa.reduce_all %2 {axis = 0 : i32} : (tensor<28x60xi1>) -> tensor<1x60xi1>
    return %0, %3 : tensor<58xf32>, tensor<1x60xi1>
  }
}
