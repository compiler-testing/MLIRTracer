module {
  func.func @main(%arg0: tensor<32x86xi32>, %arg1: tensor<1x1xi32>) -> tensor<32x1xi1> {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<32x86xi32>, tensor<1x1xi32>) -> tensor<32x86xi1>
    %1 = tosa.reduce_any %0 {axis = 1 : i32} : (tensor<32x86xi1>) -> tensor<32x1xi1>
    %2 = tosa.identity %1 : (tensor<32x1xi1>) -> tensor<32x1xi1>
    %3 = tosa.reduce_sum %2 {axis = 1 : i32} : (tensor<32x1xi1>) -> tensor<32x1xi1>
    %4 = tosa.logical_xor %3, %1 : (tensor<32x1xi1>, tensor<32x1xi1>) -> tensor<32x1xi1>
    return %4 : tensor<32x1xi1>
  }
}
