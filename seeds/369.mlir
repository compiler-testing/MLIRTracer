module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>, %arg2: tensor<7x60x5xi1>, %arg3: tensor<13x21xi1>, %arg4: tensor<29x62x37x30x42x54xf32>) -> (tensor<i8>, tensor<60x5xi32>, tensor<29x62x37x30x42x54xf32>, tensor<13x1xi1>, tensor<13x1xi1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %1 = tosa.argmax %arg2 {axis = 0 : i32} : (tensor<7x60x5xi1>) -> tensor<60x5xi32>
    %2 = tosa.reduce_all %arg3 {axis = 1 : i32} : (tensor<13x21xi1>) -> tensor<13x1xi1>
    %3 = tosa.sigmoid %arg4 : (tensor<29x62x37x30x42x54xf32>) -> tensor<29x62x37x30x42x54xf32>
    %4 = tosa.bitwise_not %2 : (tensor<13x1xi1>) -> tensor<13x1xi1>
    %5 = tosa.sub %2, %2 : (tensor<13x1xi1>, tensor<13x1xi1>) -> tensor<13x1xi1>
    return %0, %1, %3, %4, %5 : tensor<i8>, tensor<60x5xi32>, tensor<29x62x37x30x42x54xf32>, tensor<13x1xi1>, tensor<13x1xi1>
  }
}
