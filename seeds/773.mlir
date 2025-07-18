module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<70xi16>) -> (tensor<1xi16>, tensor<i1>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.reduce_product %arg2 {axis = 0 : i32} : (tensor<70xi16>) -> tensor<1xi16>
    %2 = tosa.logical_right_shift %1, %1 : (tensor<1xi16>, tensor<1xi16>) -> tensor<1xi16>
    %3 = tosa.logical_or %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %2, %3 : tensor<1xi16>, tensor<i1>
  }
}
