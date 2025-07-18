module {
  func.func @main(%arg0: tensor<9xi1>, %arg1: tensor<1xi1>, %arg2: tensor<38x39x20x58x51xf32>) -> (tensor<38x39x20x58x51xf32>, tensor<1xi1>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<9xi1>, tensor<1xi1>) -> tensor<9xi1>
    %1 = tosa.logical_and %0, %0 : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %2 = tosa.reverse %1 {axis = 0 : i32} : (tensor<9xi1>) -> tensor<9xi1>
    %3 = tosa.reduce_product %2 {axis = 0 : i32} : (tensor<9xi1>) -> tensor<1xi1>
    %4 = tosa.reverse %3 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.logical_right_shift %4, %4 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.clz %5 : (tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.rsqrt %arg2 : (tensor<38x39x20x58x51xf32>) -> tensor<38x39x20x58x51xf32>
    %8 = tosa.sub %7, %7 : (tensor<38x39x20x58x51xf32>, tensor<38x39x20x58x51xf32>) -> tensor<38x39x20x58x51xf32>
    %9 = tosa.reduce_product %6 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %10 = tosa.reverse %9 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %8, %10 : tensor<38x39x20x58x51xf32>, tensor<1xi1>
  }
}
