module {
  func.func @main(%arg0: tensor<25x63x41x51xf32>, %arg1: tensor<25x63x41x51xf32>, %arg2: tensor<91x90x23x81x42x21xi8>, %arg3: tensor<91x90x23x81x1x1xi8>, %arg4: tensor<29x38xf32>, %arg5: tensor<29x1xf32>, %arg6: tensor<24x89x69x63xi64>, %arg7: tensor<24x89x69x1xi64>) -> (tensor<91x90x23x81x42x21xi1>, tensor<25x51x63x41xi1>, tensor<29x38xi1>, tensor<24x89x69x63xi1>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<25x63x41x51xf32>, tensor<25x63x41x51xf32>) -> tensor<25x63x41x51xi1>
    %1 = tosa.abs %0 : (tensor<25x63x41x51xi1>) -> tensor<25x63x41x51xi1>
    %2 = tosa.greater %arg2, %arg3 : (tensor<91x90x23x81x42x21xi8>, tensor<91x90x23x81x1x1xi8>) -> tensor<91x90x23x81x42x21xi1>
    %3 = tosa.sub %2, %2 : (tensor<91x90x23x81x42x21xi1>, tensor<91x90x23x81x42x21xi1>) -> tensor<91x90x23x81x42x21xi1>
    %4 = tosa.logical_not %1 : (tensor<25x63x41x51xi1>) -> tensor<25x63x41x51xi1>
    %5 = tosa.bitwise_and %4, %1 : (tensor<25x63x41x51xi1>, tensor<25x63x41x51xi1>) -> tensor<25x63x41x51xi1>
    %6 = tosa.logical_xor %3, %3 : (tensor<91x90x23x81x42x21xi1>, tensor<91x90x23x81x42x21xi1>) -> tensor<91x90x23x81x42x21xi1>
    %7 = "tosa.const"() {values = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %8 = tosa.transpose %5 {perms = array<i32: 0, 3, 1, 2>} : (tensor<25x63x41x51xi1>) -> tensor<25x51x63x41xi1>
    %9 = tosa.bitwise_xor %8, %8 : (tensor<25x51x63x41xi1>, tensor<25x51x63x41xi1>) -> tensor<25x51x63x41xi1>
    %10 = tosa.reverse %9 {axis = 2 : i32} : (tensor<25x51x63x41xi1>) -> tensor<25x51x63x41xi1>
    %11 = tosa.greater %arg4, %arg5 : (tensor<29x38xf32>, tensor<29x1xf32>) -> tensor<29x38xi1>
    %12 = tosa.greater_equal %arg6, %arg7 : (tensor<24x89x69x63xi64>, tensor<24x89x69x1xi64>) -> tensor<24x89x69x63xi1>
    return %6, %10, %11, %12 : tensor<91x90x23x81x42x21xi1>, tensor<25x51x63x41xi1>, tensor<29x38xi1>, tensor<24x89x69x63xi1>
  }
}
