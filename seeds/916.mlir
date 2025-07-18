module {
  func.func @main(%arg0: tensor<42x37xi64>) -> tensor<42xi32> {
    %0 = tosa.reverse %arg0 {axis = 1 : i32} : (tensor<42x37xi64>) -> tensor<42x37xi64>
    %1 = tosa.maximum %0, %0 : (tensor<42x37xi64>, tensor<42x37xi64>) -> tensor<42x37xi64>
    %2 = tosa.logical_left_shift %1, %0 : (tensor<42x37xi64>, tensor<42x37xi64>) -> tensor<42x37xi64>
    %3 = tosa.argmax %2 {axis = 1 : i32} : (tensor<42x37xi64>) -> tensor<42xi32>
    return %3 : tensor<42xi32>
  }
}
