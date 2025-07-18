module {
  func.func @main(%arg0: tensor<95x27xi16>, %arg1: tensor<68x40x15x97x94xi64>, %arg2: tensor<1x1x1x1x94xi64>, %arg3: tensor<88x42x58x90x17xi1>, %arg4: tensor<88x42x58x1x1xi1>, %arg5: tensor<26x52x22x52x94xf32>) -> (tensor<68x40x15x97x94xi64>, tensor<6x7xi16>, tensor<88x42x58x90x17xi1>, tensor<88x42x58x90x17xi1>, tensor<26x52x22x52x94xf32>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<95x27xi16>) -> tensor<95x27xi16>
    %1 = tosa.maximum %arg1, %arg2 : (tensor<68x40x15x97x94xi64>, tensor<1x1x1x1x94xi64>) -> tensor<68x40x15x97x94xi64>
    %s_2_start = tosa.const_shape {values = dense<[ 89, 13 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_2_size = tosa.const_shape {values = dense<[ 6, 7 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.slice %0, %s_2_start, %s_2_size : (tensor<95x27xi16>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<6x7xi16>
    %3 = tosa.identity %2 : (tensor<6x7xi16>) -> tensor<6x7xi16>
    %4 = tosa.clz %3 : (tensor<6x7xi16>) -> tensor<6x7xi16>
    %5 = tosa.logical_and %arg3, %arg4 : (tensor<88x42x58x90x17xi1>, tensor<88x42x58x1x1xi1>) -> tensor<88x42x58x90x17xi1>
    %6 = tosa.floor %arg5 : (tensor<26x52x22x52x94xf32>) -> tensor<26x52x22x52x94xf32>
    %7 = tosa.bitwise_xor %5, %5 : (tensor<88x42x58x90x17xi1>, tensor<88x42x58x90x17xi1>) -> tensor<88x42x58x90x17xi1>
    %8 = tosa.logical_and %5, %5 : (tensor<88x42x58x90x17xi1>, tensor<88x42x58x90x17xi1>) -> tensor<88x42x58x90x17xi1>
    %9 = tosa.abs %7 : (tensor<88x42x58x90x17xi1>) -> tensor<88x42x58x90x17xi1>
    %10 = tosa.tanh %6 : (tensor<26x52x22x52x94xf32>) -> tensor<26x52x22x52x94xf32>
    return %1, %4, %8, %9, %10 : tensor<68x40x15x97x94xi64>, tensor<6x7xi16>, tensor<88x42x58x90x17xi1>, tensor<88x42x58x90x17xi1>, tensor<26x52x22x52x94xf32>
  }
}
