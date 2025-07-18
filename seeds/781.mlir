module {
  func.func @main(%arg0: tensor<68xi16>, %arg1: tensor<1xi16>, %arg2: tensor<i1>, %arg3: tensor<i1>) -> (tensor<i1>, tensor<136xi16>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<68xi16>, tensor<1xi16>) -> tensor<68xi16>
    %1 = tosa.logical_xor %arg2, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %2 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<68xi16>, tensor<68xi16>) -> tensor<136xi16>
    return %1, %2 : tensor<i1>, tensor<136xi16>
  }
}
