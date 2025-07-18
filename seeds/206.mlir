module {
  func.func @main(%arg0: tensor<95x40x36x86x18xi8>, %arg1: tensor<61xi16>) -> (tensor<95x40x36x86x18xi8>, tensor<i32>) {
    %0 = tosa.identity %arg0 : (tensor<95x40x36x86x18xi8>) -> tensor<95x40x36x86x18xi8>
    %1 = tosa.argmax %arg1 {axis = 0 : i32} : (tensor<61xi16>) -> tensor<i32>
    return %0, %1 : tensor<95x40x36x86x18xi8>, tensor<i32>
  }
}
