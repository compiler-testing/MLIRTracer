module {
  func.func @main(%arg0: tensor<42x90x56x86xi1>, %arg1: tensor<7x47xf32>) -> (tensor<42x90x56x1xi1>, tensor<1x47xi1>, tensor<1x47xf32>) {
    %0 = tosa.reduce_any %arg0 {axis = 3 : i32} : (tensor<42x90x56x86xi1>) -> tensor<42x90x56x1xi1>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<42x90x56x1xi1>, tensor<42x90x56x1xi1>) -> tensor<42x90x56x1xi1>
    %2 = tosa.logical_not %1 : (tensor<42x90x56x1xi1>) -> tensor<42x90x56x1xi1>
    %3 = tosa.floor %arg1 : (tensor<7x47xf32>) -> tensor<7x47xf32>
    %4 = tosa.logical_left_shift %2, %2 : (tensor<42x90x56x1xi1>, tensor<42x90x56x1xi1>) -> tensor<42x90x56x1xi1>
    %5 = tosa.equal %3, %3 : (tensor<7x47xf32>, tensor<7x47xf32>) -> tensor<7x47xi1>
    %6 = tosa.reduce_any %5 {axis = 0 : i32} : (tensor<7x47xi1>) -> tensor<1x47xi1>
    %7 = tosa.reduce_max %3 {axis = 0 : i32} : (tensor<7x47xf32>) -> tensor<1x47xf32>
    return %4, %6, %7 : tensor<42x90x56x1xi1>, tensor<1x47xi1>, tensor<1x47xf32>
  }
}
