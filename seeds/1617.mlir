module {
  func.func @main(%arg0: tensor<58x51xf32>, %arg1: tensor<17xi1>) -> (tensor<58x51xi1>, tensor<58x51xf32>, tensor<58x51xi1>, tensor<2xi1>, tensor<1xi1>, tensor<58x51xf32>) {
    %0 = tosa.tanh %arg0 : (tensor<58x51xf32>) -> tensor<58x51xf32>
    %1 = tosa.identity %0 : (tensor<58x51xf32>) -> tensor<58x51xf32>
    %2 = tosa.logical_not %arg1 : (tensor<17xi1>) -> tensor<17xi1>
    %3 = tosa.floor %1 : (tensor<58x51xf32>) -> tensor<58x51xf32>
    %4 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<17xi1>) -> tensor<1xi1>
    %5 = tosa.bitwise_and %4, %4 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.greater_equal %0, %0 : (tensor<58x51xf32>, tensor<58x51xf32>) -> tensor<58x51xi1>
    %7 = tosa.sub %5, %4 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %8 = tosa.minimum %0, %3 : (tensor<58x51xf32>, tensor<58x51xf32>) -> tensor<58x51xf32>
    %9 = tosa.greater_equal %3, %3 : (tensor<58x51xf32>, tensor<58x51xf32>) -> tensor<58x51xi1>
    %10 = tosa.concat %7, %7 {axis = 0 : i32} : (tensor<1xi1>, tensor<1xi1>) -> tensor<2xi1>
    %11 = tosa.bitwise_not %10 : (tensor<2xi1>) -> tensor<2xi1>
    %12 = tosa.reverse %4 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %13 = tosa.bitwise_not %12 : (tensor<1xi1>) -> tensor<1xi1>
    %14 = tosa.rsqrt %0 : (tensor<58x51xf32>) -> tensor<58x51xf32>
    %15 = tosa.rsqrt %14 : (tensor<58x51xf32>) -> tensor<58x51xf32>
    return %6, %8, %9, %11, %13, %15 : tensor<58x51xi1>, tensor<58x51xf32>, tensor<58x51xi1>, tensor<2xi1>, tensor<1xi1>, tensor<58x51xf32>
  }
}
