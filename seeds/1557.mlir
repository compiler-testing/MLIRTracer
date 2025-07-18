module {
  func.func @main(%arg0: tensor<88x73x57xi64>, %arg1: tensor<64x43xi1>, %arg2: tensor<1x1xi1>) -> (tensor<88x73x57xi1>, tensor<1x43xi1>, tensor<1x1xi1>) {
    %0 = tosa.identity %arg0 : (tensor<88x73x57xi64>) -> tensor<88x73x57xi64>
    %1 = tosa.logical_xor %arg1, %arg2 : (tensor<64x43xi1>, tensor<1x1xi1>) -> tensor<64x43xi1>
    %2 = tosa.clamp %0 {min_val = -51 : i64, max_val = 36 : i64} : (tensor<88x73x57xi64>) -> tensor<88x73x57xi64>
    %3 = tosa.reduce_all %1 {axis = 0 : i32} : (tensor<64x43xi1>) -> tensor<1x43xi1>
    %t_4 = tosa.const_shape {values = dense<[ 1, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %4 = tosa.tile %1, %t_4 : (tensor<64x43xi1>, !tosa.shape<2>) -> tensor<64x129xi1>
    %5 = tosa.bitwise_xor %4, %4 : (tensor<64x129xi1>, tensor<64x129xi1>) -> tensor<64x129xi1>
    %6 = tosa.greater_equal %2, %2 : (tensor<88x73x57xi64>, tensor<88x73x57xi64>) -> tensor<88x73x57xi1>
    %7 = tosa.bitwise_xor %3, %3 : (tensor<1x43xi1>, tensor<1x43xi1>) -> tensor<1x43xi1>
    %8 = tosa.reduce_all %5 {axis = 0 : i32} : (tensor<64x129xi1>) -> tensor<1x129xi1>
    %9 = tosa.reduce_all %8 {axis = 1 : i32} : (tensor<1x129xi1>) -> tensor<1x1xi1>
    return %6, %7, %9 : tensor<88x73x57xi1>, tensor<1x43xi1>, tensor<1x1xi1>
  }
}
