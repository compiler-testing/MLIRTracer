module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<93x13x1x53xf32>) -> (tensor<i32>, tensor<93x13x1x1xf32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = tosa.bitwise_xor %1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = tosa.reduce_sum %arg2 {axis = 3 : i32} : (tensor<93x13x1x53xf32>) -> tensor<93x13x1x1xf32>
    %4 = tosa.sigmoid %3 : (tensor<93x13x1x1xf32>) -> tensor<93x13x1x1xf32>
    return %2, %4 : tensor<i32>, tensor<93x13x1x1xf32>
  }
}
