module {
  func.func @main(%arg0: tensor<20x49x46x75x11xi32>, %arg1: tensor<1x49x1x1x11xi32>) -> tensor<20x49x46x75x11xi32> {
    %0 = tosa.add %arg0, %arg1 : (tensor<20x49x46x75x11xi32>, tensor<1x49x1x1x11xi32>) -> tensor<20x49x46x75x11xi32>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<20x49x46x75x11xi32>, tensor<20x49x46x75x11xi32>) -> tensor<20x49x46x75x11xi32>
    return %1 : tensor<20x49x46x75x11xi32>
  }
}
