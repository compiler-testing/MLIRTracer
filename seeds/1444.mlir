module {
  func.func @main(%arg0: tensor<51x18x47xi16>, %arg1: tensor<2x26x85x63xi1>, %arg2: tensor<2x1x1x63xi1>, %arg3: tensor<91x89x7xi64>, %arg4: tensor<1x1x1xi64>, %arg5: tensor<45x50x3x4x57xf32>) -> (tensor<846x1xi16>, tensor<91x89x7xi64>, tensor<2x26x85x63xi1>, tensor<14x39x255x2xi1>, tensor<45x50x3x4x57xi1>, tensor<2x26x85x63xi1>, tensor<45x50x3x4x57xi1>) {
    %r_0 = tosa.const_shape {values = dense<[ 846, 51 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<51x18x47xi16>, !tosa.shape<2>) -> tensor<846x51xi16>
    %1 = tosa.reduce_product %0 {axis = 1 : i32} : (tensor<846x51xi16>) -> tensor<846x1xi16>
    %2 = tosa.logical_and %arg1, %arg2 : (tensor<2x26x85x63xi1>, tensor<2x1x1x63xi1>) -> tensor<2x26x85x63xi1>
    %3 = tosa.logical_xor %2, %2 : (tensor<2x26x85x63xi1>, tensor<2x26x85x63xi1>) -> tensor<2x26x85x63xi1>
    %4 = tosa.minimum %arg3, %arg4 : (tensor<91x89x7xi64>, tensor<1x1x1xi64>) -> tensor<91x89x7xi64>
    %5 = tosa.bitwise_xor %2, %3 : (tensor<2x26x85x63xi1>, tensor<2x26x85x63xi1>) -> tensor<2x26x85x63xi1>
    %6 = tosa.sigmoid %arg5 : (tensor<45x50x3x4x57xf32>) -> tensor<45x50x3x4x57xf32>
    %r_7 = tosa.const_shape {values = dense<[ 14, 39, 255, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %7 = tosa.reshape %2, %r_7 : (tensor<2x26x85x63xi1>, !tosa.shape<4>) -> tensor<14x39x255x2xi1>
    %8 = tosa.equal %6, %6 : (tensor<45x50x3x4x57xf32>, tensor<45x50x3x4x57xf32>) -> tensor<45x50x3x4x57xi1>
    %9 = tosa.bitwise_and %2, %3 : (tensor<2x26x85x63xi1>, tensor<2x26x85x63xi1>) -> tensor<2x26x85x63xi1>
    %10 = tosa.greater_equal %6, %6 : (tensor<45x50x3x4x57xf32>, tensor<45x50x3x4x57xf32>) -> tensor<45x50x3x4x57xi1>
    return %1, %4, %5, %7, %8, %9, %10 : tensor<846x1xi16>, tensor<91x89x7xi64>, tensor<2x26x85x63xi1>, tensor<14x39x255x2xi1>, tensor<45x50x3x4x57xi1>, tensor<2x26x85x63xi1>, tensor<45x50x3x4x57xi1>
  }
}
