module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>, %arg2: tensor<62x4xi1>) -> (tensor<62x1xi1>, tensor<i8>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %1 = tosa.reduce_all %arg2 {axis = 1 : i32} : (tensor<62x4xi1>) -> tensor<62x1xi1>
    %2 = tosa.bitwise_or %0, %0 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    return %1, %2 : tensor<62x1xi1>, tensor<i8>
  }
}
