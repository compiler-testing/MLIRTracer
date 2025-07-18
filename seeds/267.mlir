module {
  func.func @main(%arg0: tensor<21x49xi64>, %arg1: tensor<1x1xi64>, %arg2: tensor<31x18xf32>) -> (tensor<49xi32>, tensor<31x18xf32>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<21x49xi64>, tensor<1x1xi64>) -> tensor<21x49xi64>
    %1 = tosa.greater %0, %0 : (tensor<21x49xi64>, tensor<21x49xi64>) -> tensor<21x49xi1>
    %2 = tosa.argmax %1 {axis = 0 : i32} : (tensor<21x49xi1>) -> tensor<49xi32>
    %3 = tosa.reciprocal %arg2 : (tensor<31x18xf32>) -> tensor<31x18xf32>
    return %2, %3 : tensor<49xi32>, tensor<31x18xf32>
  }
}
