module {
  func.func @main(%arg0: tensor<70xi8>) -> tensor<i32> {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<70xi8>) -> tensor<1xi8>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %2 = tosa.clz %1 : (tensor<1xi8>) -> tensor<1xi8>
    %3 = tosa.argmax %2 {axis = 0 : i32} : (tensor<1xi8>) -> tensor<i32>
    return %3 : tensor<i32>
  }
}
