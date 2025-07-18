module {
  func.func @main(%arg0: tensor<49x46x71x40x44xi32>) -> tensor<49x46x71x40x44xi32> {
    %0 = tosa.bitwise_not %arg0 : (tensor<49x46x71x40x44xi32>) -> tensor<49x46x71x40x44xi32>
    %1 = tosa.maximum %0, %0 : (tensor<49x46x71x40x44xi32>, tensor<49x46x71x40x44xi32>) -> tensor<49x46x71x40x44xi32>
    return %1 : tensor<49x46x71x40x44xi32>
  }
}
