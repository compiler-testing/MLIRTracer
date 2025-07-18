module {
  func.func @main(%arg0: tensor<49x44x29x45xf32>, %arg1: tensor<45x13x90xi32>, %arg2: tensor<45x1x90xi32>) -> (tensor<45x13x90xi32>, tensor<406x13860xf32>) {
    %0 = tosa.identity %arg0 : (tensor<49x44x29x45xf32>) -> tensor<49x44x29x45xf32>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<45x13x90xi32>, tensor<45x1x90xi32>) -> tensor<45x13x90xi32>
    %2 = tosa.sub %1, %1 : (tensor<45x13x90xi32>, tensor<45x13x90xi32>) -> tensor<45x13x90xi32>
    %3 = tosa.sub %0, %0 : (tensor<49x44x29x45xf32>, tensor<49x44x29x45xf32>) -> tensor<49x44x29x45xf32>
    %4 = tosa.concat %3, %3 {axis = 3 : i32} : (tensor<49x44x29x45xf32>, tensor<49x44x29x45xf32>) -> tensor<49x44x29x90xf32>
    %5 = tosa.identity %2 : (tensor<45x13x90xi32>) -> tensor<45x13x90xi32>
    %6 = tosa.reciprocal %4 : (tensor<49x44x29x90xf32>) -> tensor<49x44x29x90xf32>
    %7 = tosa.pow %6, %6 : (tensor<49x44x29x90xf32>, tensor<49x44x29x90xf32>) -> tensor<49x44x29x90xf32>
    %r_8 = tosa.const_shape {values = dense<[ 406, 13860 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %8 = tosa.reshape %7, %r_8 : (tensor<49x44x29x90xf32>, !tosa.shape<2>) -> tensor<406x13860xf32>
    return %5, %8 : tensor<45x13x90xi32>, tensor<406x13860xf32>
  }
}
