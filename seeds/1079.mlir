module {
  func.func @main(%arg0: tensor<29x8x67x11xi32>, %arg1: tensor<1x1x1x11xi32>) -> tensor<29x8x67x1xi32> {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<29x8x67x11xi32>, tensor<1x1x1x11xi32>) -> tensor<29x8x67x11xi32>
    %1 = tosa.reduce_sum %0 {axis = 3 : i32} : (tensor<29x8x67x11xi32>) -> tensor<29x8x67x1xi32>
    %2 = tosa.bitwise_xor %1, %1 : (tensor<29x8x67x1xi32>, tensor<29x8x67x1xi32>) -> tensor<29x8x67x1xi32>
    return %2 : tensor<29x8x67x1xi32>
  }
}
