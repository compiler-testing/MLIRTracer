module {
  func.func @main(%arg0: tensor<67x62x79x15xi1>, %arg1: tensor<1x1x1x15xi1>) -> tensor<1x79x15xi1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<67x62x79x15xi1>, tensor<1x1x1x15xi1>) -> tensor<67x62x79x15xi1>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<67x62x79x15xi1>) -> tensor<62x79x15xi32>
    %2 = tosa.greater %1, %1 : (tensor<62x79x15xi32>, tensor<62x79x15xi32>) -> tensor<62x79x15xi1>
    %3 = tosa.reduce_sum %2 {axis = 0 : i32} : (tensor<62x79x15xi1>) -> tensor<1x79x15xi1>
    return %3 : tensor<1x79x15xi1>
  }
}
