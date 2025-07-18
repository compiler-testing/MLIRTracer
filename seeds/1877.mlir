module {
  func.func @main(%arg0: tensor<25x88x20x22xi1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (tensor<25x1x20x22xi1>, tensor<i32>) {
    %0 = tosa.reduce_product %arg0 {axis = 1 : i32} : (tensor<25x88x20x22xi1>) -> tensor<25x1x20x22xi1>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0, %1 : tensor<25x1x20x22xi1>, tensor<i32>
  }
}
