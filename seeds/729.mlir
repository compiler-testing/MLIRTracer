module {
  func.func @main(%arg0: tensor<32xi16>, %arg1: tensor<65x56xf32>, %arg2: tensor<65x56xf32>) -> (tensor<1xi16>, tensor<65x56xi1>) {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<32xi16>) -> tensor<1xi16>
    %1 = tosa.minimum %arg1, %arg2 : (tensor<65x56xf32>, tensor<65x56xf32>) -> tensor<65x56xf32>
    %2 = tosa.equal %1, %1 : (tensor<65x56xf32>, tensor<65x56xf32>) -> tensor<65x56xi1>
    return %0, %2 : tensor<1xi16>, tensor<65x56xi1>
  }
}
