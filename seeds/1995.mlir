module {
  func.func @main(%arg0: tensor<17x33xi1>, %arg1: tensor<92x50x91x92xi32>, %arg2: tensor<92x1x91x1xi32>, %arg3: tensor<33x42x6x50x66xf32>) -> (tensor<1x1xi1>, tensor<33x42x6x50x66xi1>, tensor<33x42x6x50x66xf32>, tensor<92x50x91x92xi32>) {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<17x33xi1>) -> tensor<1x33xi1>
    %1 = tosa.reduce_product %0 {axis = 1 : i32} : (tensor<1x33xi1>) -> tensor<1x1xi1>
    %2 = tosa.intdiv %arg1, %arg2 : (tensor<92x50x91x92xi32>, tensor<92x1x91x1xi32>) -> tensor<92x50x91x92xi32>
    %3 = tosa.add %1, %1 : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    %4 = tosa.log %arg3 : (tensor<33x42x6x50x66xf32>) -> tensor<33x42x6x50x66xf32>
    %5 = tosa.greater %4, %4 : (tensor<33x42x6x50x66xf32>, tensor<33x42x6x50x66xf32>) -> tensor<33x42x6x50x66xi1>
    %6 = tosa.log %4 : (tensor<33x42x6x50x66xf32>) -> tensor<33x42x6x50x66xf32>
    %7 = tosa.logical_right_shift %2, %2 : (tensor<92x50x91x92xi32>, tensor<92x50x91x92xi32>) -> tensor<92x50x91x92xi32>
    %8 = tosa.logical_right_shift %2, %7 : (tensor<92x50x91x92xi32>, tensor<92x50x91x92xi32>) -> tensor<92x50x91x92xi32>
    return %3, %5, %6, %8 : tensor<1x1xi1>, tensor<33x42x6x50x66xi1>, tensor<33x42x6x50x66xf32>, tensor<92x50x91x92xi32>
  }
}
