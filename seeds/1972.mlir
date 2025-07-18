module {
  func.func @main(%arg0: tensor<100x6x32x100xi16>) -> tensor<100x6x1x100xi16> {
    %0 = tosa.reduce_sum %arg0 {axis = 2 : i32} : (tensor<100x6x32x100xi16>) -> tensor<100x6x1x100xi16>
    return %0 : tensor<100x6x1x100xi16>
  }
}
