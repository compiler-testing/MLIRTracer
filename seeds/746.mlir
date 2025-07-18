module {
  func.func @main(%arg0: tensor<4xi8>) -> tensor<1xi8> {
    %0 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 0>} : (tensor<4xi8>) -> tensor<4xi8>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<4xi8>) -> tensor<1xi8>
    %3 = tosa.bitwise_xor %2, %2 : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    return %3 : tensor<1xi8>
  }
}
