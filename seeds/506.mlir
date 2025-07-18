module {
  func.func @main(%arg0: tensor<59xf32>, %arg1: tensor<1xf32>, %arg2: tensor<25xf32>) -> (tensor<59xi1>, tensor<25xf32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<59xf32>, tensor<1xf32>) -> tensor<59xi1>
    %1 = tosa.logical_not %0 : (tensor<59xi1>) -> tensor<59xi1>
    %2 = tosa.reciprocal %arg2 : (tensor<25xf32>) -> tensor<25xf32>
    %3 = tosa.reciprocal %2 : (tensor<25xf32>) -> tensor<25xf32>
    %4 = tosa.minimum %3, %3 : (tensor<25xf32>, tensor<25xf32>) -> tensor<25xf32>
    %5 = tosa.log %4 : (tensor<25xf32>) -> tensor<25xf32>
    return %1, %5 : tensor<59xi1>, tensor<25xf32>
  }
}
