module {
  func.func @main(%arg0: tensor<32x30x79x31xf32>) -> tensor<30x79x31xi1> {
    %0 = tosa.rsqrt %arg0 : (tensor<32x30x79x31xf32>) -> tensor<32x30x79x31xf32>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<32x30x79x31xf32>) -> tensor<30x79x31xi32>
    %2 = tosa.intdiv %1, %1 : (tensor<30x79x31xi32>, tensor<30x79x31xi32>) -> tensor<30x79x31xi32>
    %3 = tosa.equal %2, %1 : (tensor<30x79x31xi32>, tensor<30x79x31xi32>) -> tensor<30x79x31xi1>
    return %3 : tensor<30x79x31xi1>
  }
}
