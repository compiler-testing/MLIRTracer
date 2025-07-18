module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>, %arg2: tensor<66xf32>) -> (tensor<i1>, tensor<1xf32>, tensor<i32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i1>
    %1 = tosa.sub %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %2 = tosa.tanh %arg2 : (tensor<66xf32>) -> tensor<66xf32>
    %3 = tosa.sub %1, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<66xf32>) -> tensor<1xf32>
    %5 = tosa.pow %2, %2 : (tensor<66xf32>, tensor<66xf32>) -> tensor<66xf32>
    %6 = tosa.argmax %5 {axis = 0 : i32} : (tensor<66xf32>) -> tensor<i32>
    return %3, %4, %6 : tensor<i1>, tensor<1xf32>, tensor<i32>
  }
}
