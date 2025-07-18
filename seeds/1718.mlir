module {
  func.func @main(%arg0: tensor<20x87x34x8x96xi64>, %arg1: tensor<1x87x34x8x96xi64>, %arg2: tensor<1x57xi32>) -> (tensor<1x57xi32>, tensor<20x87x34x8x96xi64>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<20x87x34x8x96xi64>, tensor<1x87x34x8x96xi64>) -> tensor<20x87x34x8x96xi64>
    %1 = tosa.reverse %arg2 {axis = 1 : i32} : (tensor<1x57xi32>) -> tensor<1x57xi32>
    %2 = tosa.abs %0 : (tensor<20x87x34x8x96xi64>) -> tensor<20x87x34x8x96xi64>
    return %1, %2 : tensor<1x57xi32>, tensor<20x87x34x8x96xi64>
  }
}
