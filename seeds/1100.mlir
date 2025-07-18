module {
  func.func @main(%arg0: tensor<66x46x24xi1>, %arg1: tensor<98x55x98x66x80xi32>, %arg2: tensor<98x55x98x66x80xi32>) -> (tensor<66x46x1xi1>, tensor<98x55x98x66x80xi32>) {
    %0 = tosa.reduce_all %arg0 {axis = 2 : i32} : (tensor<66x46x24xi1>) -> tensor<66x46x1xi1>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<98x55x98x66x80xi32>, tensor<98x55x98x66x80xi32>) -> tensor<98x55x98x66x80xi32>
    %2 = tosa.bitwise_and %1, %1 : (tensor<98x55x98x66x80xi32>, tensor<98x55x98x66x80xi32>) -> tensor<98x55x98x66x80xi32>
    %3 = tosa.bitwise_and %0, %0 : (tensor<66x46x1xi1>, tensor<66x46x1xi1>) -> tensor<66x46x1xi1>
    %4 = tosa.intdiv %2, %1 : (tensor<98x55x98x66x80xi32>, tensor<98x55x98x66x80xi32>) -> tensor<98x55x98x66x80xi32>
    %5 = tosa.bitwise_and %4, %1 : (tensor<98x55x98x66x80xi32>, tensor<98x55x98x66x80xi32>) -> tensor<98x55x98x66x80xi32>
    return %3, %5 : tensor<66x46x1xi1>, tensor<98x55x98x66x80xi32>
  }
}
