module {
  func.func @main(%arg0: tensor<50xi8>, %arg1: tensor<63xi8>) -> tensor<113xi8> {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<50xi8>, tensor<63xi8>) -> tensor<113xi8>
    %1 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 0>} : (tensor<113xi8>) -> tensor<113xi8>
    return %2 : tensor<113xi8>
  }
}
