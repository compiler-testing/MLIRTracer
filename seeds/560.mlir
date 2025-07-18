module {
  func.func @main(%arg0: tensor<18x83x49xi64>, %arg1: tensor<26x96x86x53x74x53xf32>) -> (tensor<18x83xi1>, tensor<26x96x86x53x74x53xf32>) {
    %0 = tosa.argmax %arg0 {axis = 2 : i32} : (tensor<18x83x49xi64>) -> tensor<18x83xi32>
    %1 = tosa.bitwise_and %0, %0 : (tensor<18x83xi32>, tensor<18x83xi32>) -> tensor<18x83xi32>
    %2 = tosa.sigmoid %arg1 : (tensor<26x96x86x53x74x53xf32>) -> tensor<26x96x86x53x74x53xf32>
    %3 = tosa.greater %1, %0 : (tensor<18x83xi32>, tensor<18x83xi32>) -> tensor<18x83xi1>
    %4 = tosa.reciprocal %2 : (tensor<26x96x86x53x74x53xf32>) -> tensor<26x96x86x53x74x53xf32>
    return %3, %4 : tensor<18x83xi1>, tensor<26x96x86x53x74x53xf32>
  }
}
