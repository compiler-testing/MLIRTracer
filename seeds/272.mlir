module {
  func.func @main(%arg0: tensor<74xi64>, %arg1: tensor<1xi64>) -> (tensor<i32>, tensor<74xi64>, tensor<1xi64>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<74xi64>, tensor<1xi64>) -> tensor<74xi64>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<74xi64>) -> tensor<i32>
    %2 = tosa.maximum %0, %0 : (tensor<74xi64>, tensor<74xi64>) -> tensor<74xi64>
    %3 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<74xi64>) -> tensor<1xi64>
    return %1, %2, %3 : tensor<i32>, tensor<74xi64>, tensor<1xi64>
  }
}
