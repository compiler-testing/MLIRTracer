module {
  func.func @main(%arg0: tensor<18x65x74x83x48x38xi32>, %arg1: tensor<1x1x1x83x1x1xi32>, %arg2: tensor<15xf32>) -> (tensor<18x65x74x83x48x38xi32>, tensor<15xf32>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<18x65x74x83x48x38xi32>, tensor<1x1x1x83x1x1xi32>) -> tensor<18x65x74x83x48x38xi32>
    %1 = tosa.tanh %arg2 : (tensor<15xf32>) -> tensor<15xf32>
    %2 = tosa.bitwise_not %0 : (tensor<18x65x74x83x48x38xi32>) -> tensor<18x65x74x83x48x38xi32>
    %3 = tosa.sub %1, %1 : (tensor<15xf32>, tensor<15xf32>) -> tensor<15xf32>
    %4 = tosa.ceil %3 : (tensor<15xf32>) -> tensor<15xf32>
    return %2, %4 : tensor<18x65x74x83x48x38xi32>, tensor<15xf32>
  }
}
