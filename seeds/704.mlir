module {
  func.func @main(%arg0: tensor<28xi32>, %arg1: tensor<22x85x72x11xi1>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<22x1x72x11xi1>, tensor<i1>, tensor<i32>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<28xi32>) -> tensor<i32>
    %1 = tosa.reduce_any %arg1 {axis = 1 : i32} : (tensor<22x85x72x11xi1>) -> tensor<22x1x72x11xi1>
    %2 = tosa.logical_left_shift %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = tosa.logical_left_shift %1, %1 : (tensor<22x1x72x11xi1>, tensor<22x1x72x11xi1>) -> tensor<22x1x72x11xi1>
    %4 = tosa.pow %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %5 = tosa.greater_equal %4, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = tosa.logical_right_shift %2, %2 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %7 = tosa.bitwise_xor %6, %6 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %3, %5, %7 : tensor<22x1x72x11xi1>, tensor<i1>, tensor<i32>
  }
}
