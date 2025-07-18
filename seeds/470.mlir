module {
  func.func @main(%arg0: tensor<21x41x43x82xi32>) -> tensor<21x41x43xi1> {
    %0 = tosa.argmax %arg0 {axis = 3 : i32} : (tensor<21x41x43x82xi32>) -> tensor<21x41x43xi32>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<21x41x43xi32>, tensor<21x41x43xi32>) -> tensor<21x41x43xi32>
    %2 = tosa.bitwise_or %1, %0 : (tensor<21x41x43xi32>, tensor<21x41x43xi32>) -> tensor<21x41x43xi32>
    %3 = tosa.greater %2, %0 : (tensor<21x41x43xi32>, tensor<21x41x43xi32>) -> tensor<21x41x43xi1>
    return %3 : tensor<21x41x43xi1>
  }
}
