module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>, %arg2: tensor<14x71x21xi64>, %arg3: tensor<14x1x1xi64>) -> (tensor<14x71x21xi1>, tensor<i1>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i1>
    %1 = tosa.sub %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %2 = tosa.bitwise_not %1 : (tensor<i1>) -> tensor<i1>
    %3 = tosa.clz %2 : (tensor<i1>) -> tensor<i1>
    %4 = tosa.equal %arg2, %arg3 : (tensor<14x71x21xi64>, tensor<14x1x1xi64>) -> tensor<14x71x21xi1>
    %5 = tosa.bitwise_not %4 : (tensor<14x71x21xi1>) -> tensor<14x71x21xi1>
    %6 = tosa.logical_not %3 : (tensor<i1>) -> tensor<i1>
    return %5, %6 : tensor<14x71x21xi1>, tensor<i1>
  }
}
