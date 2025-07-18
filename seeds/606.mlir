module {
  func.func @main(%arg0: tensor<i16>, %arg1: tensor<i16>, %arg2: tensor<91xi64>, %arg3: tensor<1xi64>) -> (tensor<i16>, tensor<91xi1>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %1 = tosa.greater_equal %arg2, %arg3 : (tensor<91xi64>, tensor<1xi64>) -> tensor<91xi1>
    return %0, %1 : tensor<i16>, tensor<91xi1>
  }
}
