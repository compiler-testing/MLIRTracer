module {
  func.func @main(%arg0: tensor<93x86x89xi64>, %arg1: tensor<93x86x1xi64>, %arg2: tensor<54x96xi1>, %arg3: tensor<1x1xi1>, %arg4: tensor<20x90x86x33x71xf32>) -> (tensor<54x96xi1>, tensor<86x1xi32>, tensor<20x90x86x33x71xf32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<93x86x89xi64>, tensor<93x86x1xi64>) -> tensor<93x86x89xi64>
    %1 = tosa.reduce_sum %0 {axis = 2 : i32} : (tensor<93x86x89xi64>) -> tensor<93x86x1xi64>
    %2 = tosa.logical_xor %arg2, %arg3 : (tensor<54x96xi1>, tensor<1x1xi1>) -> tensor<54x96xi1>
    %3 = tosa.argmax %1 {axis = 0 : i32} : (tensor<93x86x1xi64>) -> tensor<86x1xi32>
    %4 = tosa.intdiv %3, %3 : (tensor<86x1xi32>, tensor<86x1xi32>) -> tensor<86x1xi32>
    %5 = tosa.log %arg4 : (tensor<20x90x86x33x71xf32>) -> tensor<20x90x86x33x71xf32>
    %6 = tosa.maximum %4, %4 : (tensor<86x1xi32>, tensor<86x1xi32>) -> tensor<86x1xi32>
    %7 = tosa.bitwise_xor %2, %2 : (tensor<54x96xi1>, tensor<54x96xi1>) -> tensor<54x96xi1>
    %8 = tosa.abs %6 : (tensor<86x1xi32>) -> tensor<86x1xi32>
    %9 = tosa.add %8, %8 : (tensor<86x1xi32>, tensor<86x1xi32>) -> tensor<86x1xi32>
    %10 = tosa.reciprocal %5 : (tensor<20x90x86x33x71xf32>) -> tensor<20x90x86x33x71xf32>
    return %7, %9, %10 : tensor<54x96xi1>, tensor<86x1xi32>, tensor<20x90x86x33x71xf32>
  }
}
