module {
  func.func @main(%arg0: tensor<97xf32>, %arg1: tensor<4x47x5x19x89xi8>) -> (tensor<97xf32>, tensor<4x47x5x19x89xi8>) {
    %0 = tosa.identity %arg0 : (tensor<97xf32>) -> tensor<97xf32>
    %1 = tosa.add %0, %0 : (tensor<97xf32>, tensor<97xf32>) -> tensor<97xf32>
    %2 = tosa.exp %1 : (tensor<97xf32>) -> tensor<97xf32>
    %3 = tosa.tanh %2 : (tensor<97xf32>) -> tensor<97xf32>
    %4 = tosa.bitwise_not %arg1 : (tensor<4x47x5x19x89xi8>) -> tensor<4x47x5x19x89xi8>
    %5 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %6 = tosa.transpose %3 {perms = array<i32: 0>} : (tensor<97xf32>) -> tensor<97xf32>
    %7 = tosa.bitwise_and %4, %4 : (tensor<4x47x5x19x89xi8>, tensor<4x47x5x19x89xi8>) -> tensor<4x47x5x19x89xi8>
    return %6, %7 : tensor<97xf32>, tensor<4x47x5x19x89xi8>
  }
}
