module {
  func.func @main(%arg0: tensor<20x59x51xi16>, %arg1: tensor<45x31xi1>, %arg2: tensor<53x70x57x64xf32>) -> (tensor<20x59x1xi16>, tensor<45x1xi1>, tensor<53x70x57x64xf32>, tensor<53x70x57x64xi1>, tensor<53x70x57x64xi1>, tensor<53x70x57x64xi1>) {
    %0 = tosa.abs %arg0 : (tensor<20x59x51xi16>) -> tensor<20x59x51xi16>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<20x59x51xi16>, tensor<20x59x51xi16>) -> tensor<20x59x51xi16>
    %2 = tosa.abs %1 : (tensor<20x59x51xi16>) -> tensor<20x59x51xi16>
    %3 = tosa.logical_not %arg1 : (tensor<45x31xi1>) -> tensor<45x31xi1>
    %4 = tosa.floor %arg2 : (tensor<53x70x57x64xf32>) -> tensor<53x70x57x64xf32>
    %5 = tosa.reduce_min %2 {axis = 2 : i32} : (tensor<20x59x51xi16>) -> tensor<20x59x1xi16>
    %6 = tosa.greater_equal %4, %4 : (tensor<53x70x57x64xf32>, tensor<53x70x57x64xf32>) -> tensor<53x70x57x64xi1>
    %7 = tosa.maximum %4, %4 : (tensor<53x70x57x64xf32>, tensor<53x70x57x64xf32>) -> tensor<53x70x57x64xf32>
    %8 = tosa.reduce_all %3 {axis = 1 : i32} : (tensor<45x31xi1>) -> tensor<45x1xi1>
    %9 = tosa.minimum %4, %4 : (tensor<53x70x57x64xf32>, tensor<53x70x57x64xf32>) -> tensor<53x70x57x64xf32>
    %10 = tosa.greater_equal %4, %4 : (tensor<53x70x57x64xf32>, tensor<53x70x57x64xf32>) -> tensor<53x70x57x64xi1>
    %11 = tosa.bitwise_xor %10, %6 : (tensor<53x70x57x64xi1>, tensor<53x70x57x64xi1>) -> tensor<53x70x57x64xi1>
    %12 = tosa.logical_and %10, %11 : (tensor<53x70x57x64xi1>, tensor<53x70x57x64xi1>) -> tensor<53x70x57x64xi1>
    %13 = tosa.ceil %4 : (tensor<53x70x57x64xf32>) -> tensor<53x70x57x64xf32>
    %14 = tosa.add %12, %10 : (tensor<53x70x57x64xi1>, tensor<53x70x57x64xi1>) -> tensor<53x70x57x64xi1>
    %15 = tosa.bitwise_not %12 : (tensor<53x70x57x64xi1>) -> tensor<53x70x57x64xi1>
    %16 = tosa.greater %13, %7 : (tensor<53x70x57x64xf32>, tensor<53x70x57x64xf32>) -> tensor<53x70x57x64xi1>
    return %5, %8, %9, %14, %15, %16 : tensor<20x59x1xi16>, tensor<45x1xi1>, tensor<53x70x57x64xf32>, tensor<53x70x57x64xi1>, tensor<53x70x57x64xi1>, tensor<53x70x57x64xi1>
  }
}
