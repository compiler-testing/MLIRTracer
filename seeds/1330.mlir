module {
  func.func @main(%arg0: tensor<6x37x95xf32>, %arg1: tensor<79x100xi1>, %arg2: tensor<1x100xi1>) -> (tensor<6x37xi32>, tensor<79x1xi1>) {
    %0 = tosa.argmax %arg0 {axis = 2 : i32} : (tensor<6x37x95xf32>) -> tensor<6x37xi32>
    %1 = tosa.minimum %0, %0 : (tensor<6x37xi32>, tensor<6x37xi32>) -> tensor<6x37xi32>
    %2 = tosa.bitwise_not %1 : (tensor<6x37xi32>) -> tensor<6x37xi32>
    %3 = tosa.logical_and %arg1, %arg2 : (tensor<79x100xi1>, tensor<1x100xi1>) -> tensor<79x100xi1>
    %4 = tosa.reduce_sum %3 {axis = 1 : i32} : (tensor<79x100xi1>) -> tensor<79x1xi1>
    %5 = tosa.logical_right_shift %2, %2 : (tensor<6x37xi32>, tensor<6x37xi32>) -> tensor<6x37xi32>
    %6 = tosa.logical_xor %4, %4 : (tensor<79x1xi1>, tensor<79x1xi1>) -> tensor<79x1xi1>
    return %5, %6 : tensor<6x37xi32>, tensor<79x1xi1>
  }
}
