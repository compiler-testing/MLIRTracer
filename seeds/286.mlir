module {
  func.func @main(%arg0: tensor<92x28x85xi8>, %arg1: tensor<46xf32>) -> (tensor<92x28x85xi8>, tensor<46xi1>, tensor<46xf32>) {
    %0 = tosa.clz %arg0 : (tensor<92x28x85xi8>) -> tensor<92x28x85xi8>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<92x28x85xi8>) -> tensor<92x28x85xi8>
    %2 = tosa.reverse %1 {axis = 2 : i32} : (tensor<92x28x85xi8>) -> tensor<92x28x85xi8>
    %3 = tosa.ceil %arg1 : (tensor<46xf32>) -> tensor<46xf32>
    %4 = tosa.rsqrt %3 : (tensor<46xf32>) -> tensor<46xf32>
    %5 = tosa.equal %4, %4 : (tensor<46xf32>, tensor<46xf32>) -> tensor<46xi1>
    %6 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %7 = tosa.transpose %3 {perms = array<i32: 0>} : (tensor<46xf32>) -> tensor<46xf32>
    return %2, %5, %7 : tensor<92x28x85xi8>, tensor<46xi1>, tensor<46xf32>
  }
}
