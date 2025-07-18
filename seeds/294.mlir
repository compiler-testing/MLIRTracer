module {
  func.func @main(%arg0: tensor<97x14x17xi64>, %arg1: tensor<97x1x1xi64>) -> tensor<194x14x1xi64> {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<97x14x17xi64>, tensor<97x1x1xi64>) -> tensor<97x14x17xi64>
    %1 = tosa.abs %0 : (tensor<97x14x17xi64>) -> tensor<97x14x17xi64>
    %2 = tosa.reverse %1 {axis = 0 : i32} : (tensor<97x14x17xi64>) -> tensor<97x14x17xi64>
    %3 = tosa.bitwise_not %2 : (tensor<97x14x17xi64>) -> tensor<97x14x17xi64>
    %4 = tosa.concat %3, %1 {axis = 0 : i32} : (tensor<97x14x17xi64>, tensor<97x14x17xi64>) -> tensor<194x14x17xi64>
    %5 = tosa.reduce_sum %4 {axis = 2 : i32} : (tensor<194x14x17xi64>) -> tensor<194x14x1xi64>
    return %5 : tensor<194x14x1xi64>
  }
}
