module {
  func.func @main(%arg0: tensor<7x83x56xi8>, %arg1: tensor<56xf32>, %arg2: tensor<1xi1>, %arg3: tensor<1xi1>) -> (tensor<i32>, tensor<56xf32>, tensor<7x83x1xi8>, tensor<1xi1>) {
    %0 = tosa.reduce_sum %arg0 {axis = 2 : i32} : (tensor<7x83x56xi8>) -> tensor<7x83x1xi8>
    %1 = tosa.tanh %arg1 : (tensor<56xf32>) -> tensor<56xf32>
    %2 = tosa.floor %1 : (tensor<56xf32>) -> tensor<56xf32>
    %3 = tosa.argmax %2 {axis = 0 : i32} : (tensor<56xf32>) -> tensor<i32>
    %4 = tosa.minimum %1, %2 : (tensor<56xf32>, tensor<56xf32>) -> tensor<56xf32>
    %5 = tosa.bitwise_and %0, %0 : (tensor<7x83x1xi8>, tensor<7x83x1xi8>) -> tensor<7x83x1xi8>
    %6 = tosa.logical_and %arg2, %arg3 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %3, %4, %5, %6 : tensor<i32>, tensor<56xf32>, tensor<7x83x1xi8>, tensor<1xi1>
  }
}
