module {
  func.func @main(%arg0: tensor<45x77x26x100x2x86xi32>, %arg1: tensor<45x77x26x1x1x86xi32>, %arg2: tensor<49x43x90xi1>) -> (tensor<45x77x26x100x2x86xi32>, tensor<49x43x90xi1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<45x77x26x100x2x86xi32>, tensor<45x77x26x1x1x86xi32>) -> tensor<45x77x26x100x2x86xi32>
    %1 = tosa.bitwise_and %0, %0 : (tensor<45x77x26x100x2x86xi32>, tensor<45x77x26x100x2x86xi32>) -> tensor<45x77x26x100x2x86xi32>
    %2 = tosa.reverse %arg2 {axis = 0 : i32} : (tensor<49x43x90xi1>) -> tensor<49x43x90xi1>
    %3 = tosa.abs %2 : (tensor<49x43x90xi1>) -> tensor<49x43x90xi1>
    return %1, %3 : tensor<45x77x26x100x2x86xi32>, tensor<49x43x90xi1>
  }
}
