module {
  func.func @main(%arg0: tensor<73xi64>, %arg1: tensor<73xi64>) -> tensor<1xi64> {
    %0 = tosa.sub %arg0, %arg1 : (tensor<73xi64>, tensor<73xi64>) -> tensor<73xi64>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<73xi64>, tensor<73xi64>) -> tensor<73xi64>
    %2 = tosa.reduce_product %1 {axis = 0 : i32} : (tensor<73xi64>) -> tensor<1xi64>
    return %2 : tensor<1xi64>
  }
}
