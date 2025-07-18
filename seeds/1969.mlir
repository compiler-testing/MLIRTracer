module {
  func.func @main(%arg0: tensor<60x25x66x40x91xi64>, %arg1: tensor<60x1x66x40x91xi64>, %arg2: tensor<25xi1>) -> (tensor<60x25x66x40x91xi1>, tensor<1xi1>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<60x25x66x40x91xi64>, tensor<60x1x66x40x91xi64>) -> tensor<60x25x66x40x91xi1>
    %1 = tosa.clz %0 : (tensor<60x25x66x40x91xi1>) -> tensor<60x25x66x40x91xi1>
    %2 = tosa.reduce_product %arg2 {axis = 0 : i32} : (tensor<25xi1>) -> tensor<1xi1>
    return %1, %2 : tensor<60x25x66x40x91xi1>, tensor<1xi1>
  }
}
